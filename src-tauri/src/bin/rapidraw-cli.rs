use std::collections::HashMap;
use std::io::{BufRead, Cursor};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use image::imageops::FilterType;
use image::{GenericImageView, ImageFormat};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex as TokioMutex;

use rapidraw_lib::{
    AppSettings, AppState, ThumbnailManager, ThumbnailProgressTracker, apply_all_transformations,
};
use rapidraw_lib::cache_utils::DecodedImageCache;
use rapidraw_lib::formats::is_raw_file;
use rapidraw_lib::gpu_processing::{init_gpu_context_headless, process_and_get_dynamic_image};
use rapidraw_lib::image_loader::load_base_image_from_bytes;
use rapidraw_lib::image_processing::{
    AllAdjustments, GpuContext, RenderRequest, get_all_adjustments_from_json,
};
use rapidraw_lib::lut_processing::parse_lut_file;
use rapidraw_lib::mask_generation::{MaskDefinition, generate_mask_bitmap};

fn init_logging() {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!("[{}] {}", record.level(), message))
        })
        .level(log::LevelFilter::Info)
        .chain(std::io::stderr())
        .apply()
        .ok();
}

fn make_state() -> AppState {
    AppState {
        window_setup_complete: AtomicBool::new(false),
        gpu_crash_flag_path: Mutex::new(None),
        original_image: Mutex::new(None),
        cached_preview: Mutex::new(None),
        gpu_context: Mutex::new(None),
        gpu_image_cache: Mutex::new(None),
        gpu_processor: Mutex::new(None),
        ai_state: Mutex::new(None),
        ai_init_lock: TokioMutex::new(()),
        export_task_handle: Mutex::new(None),
        hdr_result: Arc::new(Mutex::new(None)),
        panorama_result: Arc::new(Mutex::new(None)),
        denoise_result: Arc::new(Mutex::new(None)),
        indexing_task_handle: Mutex::new(None),
        lut_cache: Mutex::new(HashMap::new()),
        initial_file_path: Mutex::new(None),
        thumbnail_cancellation_token: Arc::new(AtomicBool::new(false)),
        thumbnail_progress: Mutex::new(ThumbnailProgressTracker { total: 0, completed: 0 }),
        preview_worker_tx: Mutex::new(None),
        analytics_worker_tx: Mutex::new(None),
        mask_cache: Mutex::new(HashMap::new()),
        patch_cache: Mutex::new(HashMap::new()),
        geometry_cache: Mutex::new(HashMap::new()),
        thumbnail_geometry_cache: Mutex::new(HashMap::new()),
        lens_db: Mutex::new(None),
        load_image_generation: Arc::new(AtomicUsize::new(0)),
        full_warped_cache: Mutex::new(None),
        full_transformed_cache: Mutex::new(None),
        decoded_image_cache: Mutex::new(DecodedImageCache::new(1)),
        thumbnail_manager: ThumbnailManager::new(),
    }
}

struct Args {
    input: PathBuf,
    preset: PathBuf,
    output: PathBuf,
    quality: u8,
    resize: Option<u32>,
    server: bool,
}

#[derive(Deserialize)]
struct Job {
    id: Option<Value>,
    input: PathBuf,
    preset: PathBuf,
    output: PathBuf,
    #[serde(default = "default_quality")]
    quality: u8,
    resize: Option<u32>,
}

fn default_quality() -> u8 { 90 }

#[derive(Serialize)]
#[serde(untagged)]
enum JobResult {
    Ok { ok: bool, #[serde(skip_serializing_if = "Option::is_none")] id: Option<Value>, ms: u128 },
    Err { ok: bool, #[serde(skip_serializing_if = "Option::is_none")] id: Option<Value>, error: String },
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let mut input = None;
    let mut preset = None;
    let mut output = None;
    let mut quality = 90u8;
    let mut resize = None;
    let mut server = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input" | "-i" => { i += 1; input = Some(PathBuf::from(&argv[i])); }
            "--preset" | "-p" => { i += 1; preset = Some(PathBuf::from(&argv[i])); }
            "--output" | "-o" => { i += 1; output = Some(PathBuf::from(&argv[i])); }
            "--quality" | "-q" => {
                i += 1;
                quality = argv[i].parse().map_err(|_| "Invalid --quality (1–100)")?;
            }
            "--resize" | "-r" => {
                i += 1;
                resize = Some(argv[i].parse::<u32>().map_err(|_| "Invalid --resize (pixels)")?);
            }
            "--server" | "-s" => { server = true; }
            "--help" | "-h" => { print_usage(); std::process::exit(0); }
            other => return Err(format!("Unknown argument: {other}")),
        }
        i += 1;
    }

    if server {
        return Ok(Args { input: PathBuf::new(), preset: PathBuf::new(), output: PathBuf::new(), quality, resize, server });
    }

    Ok(Args {
        input: input.ok_or("--input is required")?,
        preset: preset.ok_or("--preset is required")?,
        output: output.ok_or("--output is required")?,
        quality,
        resize,
        server,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: rapidraw-cli --input <image> --preset <preset.rrdata> --output <out.jpg> [options]
       rapidraw-cli --server

Options:
  --input,   -i  Input image path (RAW or JPEG/PNG/etc.)
  --preset,  -p  Preset JSON path (.rrdata sidecar or standalone preset file)
  --output,  -o  Output path (.jpg/.png/.webp/.tif)
  --quality, -q  JPEG/WebP quality 1–100 (default: 90)
  --resize,  -r  Resize long edge to N pixels before saving
  --server,  -s  Server mode: read JSON jobs from stdin, write results to stdout
  --help,    -h  Show this help

Set WGPU_BACKEND=vulkan|metal|dx12|gl to override GPU backend."
    );
}

fn encode_and_save(image: &image::DynamicImage, output_path: &Path, quality: u8) -> Result<(), String> {
    let ext = output_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg")
        .to_lowercase();

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);

    match ext.as_str() {
        "jpg" | "jpeg" => {
            use image::codecs::jpeg::JpegEncoder;
            image
                .write_with_encoder(JpegEncoder::new_with_quality(&mut cursor, quality))
                .map_err(|e| e.to_string())?;
        }
        "png" => image.write_to(&mut cursor, ImageFormat::Png).map_err(|e| e.to_string())?,
        "webp" => image.write_to(&mut cursor, ImageFormat::WebP).map_err(|e| e.to_string())?,
        "tif" | "tiff" => image.write_to(&mut cursor, ImageFormat::Tiff).map_err(|e| e.to_string())?,
        other => return Err(format!("Unsupported output format: .{other}")),
    }

    drop(cursor);
    std::fs::write(output_path, bytes).map_err(|e| e.to_string())
}

struct Timing {
    label: &'static str,
    ms: u128,
}

fn ms(t: Instant) -> u128 {
    t.elapsed().as_millis()
}

fn process_image(
    input: &Path,
    preset: &Path,
    output: &Path,
    quality: u8,
    resize: Option<u32>,
    context: &GpuContext,
    state: &AppState,
) -> Result<u128, String> {
    let t_total = Instant::now();

    let input_str = input.to_string_lossy().to_string();
    let is_raw = is_raw_file(&input_str);
    let image_bytes = std::fs::read(input).map_err(|e| e.to_string())?;
    let base_image = load_base_image_from_bytes(
        &image_bytes,
        &input_str,
        false,
        &AppSettings::default(),
        None,
    )
    .map_err(|e| e.to_string())?;

    let preset_json: Value = serde_json::from_str(
        &std::fs::read_to_string(preset).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Preset JSON parse error: {e}"))?;

    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(&base_image, &preset_json);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = preset_json
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let mask_bitmaps: Vec<image::ImageBuffer<image::Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset, None))
        .collect();

    let with_color_masks = mask_definitions
        .iter()
        .flat_map(|d| d.sub_masks.iter())
        .filter(|sm| sm.mask_type == "color" || sm.mask_type == "luminance")
        .count();
    if with_color_masks > 0 {
        eprintln!("Note: {with_color_masks} color/luminance mask(s) skipped (need full-res source)");
    }

    let lut = preset_json["lutPath"].as_str().and_then(|p| {
        match parse_lut_file(p) {
            Ok(lut) => Some(Arc::new(lut)),
            Err(e) => { eprintln!("Warning: could not load LUT '{p}': {e}"); None }
        }
    });

    let mut all_adjustments: AllAdjustments =
        get_all_adjustments_from_json(&preset_json, is_raw, None);
    all_adjustments.global.show_clipping = 0;

    let processed = process_and_get_dynamic_image(
        context,
        state,
        transformed_image.as_ref(),
        0,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
        },
        "cli",
    )?;

    let final_image = match resize {
        Some(long_edge) => {
            let (w, h) = processed.dimensions();
            let max_dim = w.max(h);
            if max_dim > long_edge {
                let scale = long_edge as f32 / max_dim as f32;
                let nw = (w as f32 * scale).round() as u32;
                let nh = (h as f32 * scale).round() as u32;
                processed.resize(nw, nh, FilterType::Lanczos3)
            } else {
                processed
            }
        }
        None => processed,
    };

    encode_and_save(&final_image, output, quality)?;
    Ok(t_total.elapsed().as_millis())
}

fn run(args: Args) -> Result<(), String> {
    let t_total = Instant::now();
    let mut timings: Vec<Timing> = Vec::new();

    // ── Load image ────────────────────────────────────────────────────────────
    let t = Instant::now();
    let input_str = args.input.to_string_lossy().to_string();
    let is_raw = is_raw_file(&input_str);
    let image_bytes = std::fs::read(&args.input).map_err(|e| e.to_string())?;
    let base_image = load_base_image_from_bytes(
        &image_bytes,
        &input_str,
        false,
        &AppSettings::default(),
        None,
    )
    .map_err(|e| e.to_string())?;
    timings.push(Timing { label: "decode (RAW→linear)", ms: ms(t) });

    // ── Load preset ───────────────────────────────────────────────────────────
    let t = Instant::now();
    let preset_json: Value = serde_json::from_str(
        &std::fs::read_to_string(&args.preset).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("Preset JSON parse error: {e}"))?;
    timings.push(Timing { label: "preset parse", ms: ms(t) });

    // ── Geometry transform + masks ────────────────────────────────────────────
    let t = Instant::now();
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(&base_image, &preset_json);
    let (img_w, img_h) = transformed_image.dimensions();

    let mask_definitions: Vec<MaskDefinition> = preset_json
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let mask_bitmaps: Vec<image::ImageBuffer<image::Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset, None))
        .collect();

    let with_color_masks = mask_definitions
        .iter()
        .flat_map(|d| d.sub_masks.iter())
        .filter(|sm| sm.mask_type == "color" || sm.mask_type == "luminance")
        .count();
    if with_color_masks > 0 {
        eprintln!("Note: {with_color_masks} color/luminance mask(s) skipped (need full-res source)");
    }
    timings.push(Timing { label: "geometry + mask raster", ms: ms(t) });

    // ── LUT ───────────────────────────────────────────────────────────────────
    let lut = preset_json["lutPath"].as_str().and_then(|p| {
        match parse_lut_file(p) {
            Ok(lut) => Some(Arc::new(lut)),
            Err(e) => { eprintln!("Warning: could not load LUT '{p}': {e}"); None }
        }
    });

    // ── Adjustments ───────────────────────────────────────────────────────────
    let mut all_adjustments: AllAdjustments =
        get_all_adjustments_from_json(&preset_json, is_raw, None);
    all_adjustments.global.show_clipping = 0;

    // ── GPU init ──────────────────────────────────────────────────────────────
    let t = Instant::now();
    let state = make_state();
    let context: GpuContext = init_gpu_context_headless(&state)?;
    timings.push(Timing { label: "GPU init", ms: ms(t) });

    // ── GPU process ───────────────────────────────────────────────────────────
    let t = Instant::now();
    let processed = process_and_get_dynamic_image(
        &context,
        &state,
        transformed_image.as_ref(),
        0,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &mask_bitmaps,
            lut,
            roi: None,
        },
        "cli",
    )?;
    timings.push(Timing { label: "GPU process + readback", ms: ms(t) });

    // ── Resize ────────────────────────────────────────────────────────────────
    let t = Instant::now();
    let final_image = match args.resize {
        Some(long_edge) => {
            let (w, h) = processed.dimensions();
            let max_dim = w.max(h);
            if max_dim > long_edge {
                let scale = long_edge as f32 / max_dim as f32;
                let nw = (w as f32 * scale).round() as u32;
                let nh = (h as f32 * scale).round() as u32;
                processed.resize(nw, nh, FilterType::Lanczos3)
            } else {
                processed
            }
        }
        None => processed,
    };
    let resize_ms = ms(t);
    if args.resize.is_some() {
        timings.push(Timing { label: "resize (Lanczos3)", ms: resize_ms });
    }

    // ── Encode + save ─────────────────────────────────────────────────────────
    let t = Instant::now();
    encode_and_save(&final_image, &args.output, args.quality)?;
    timings.push(Timing { label: "encode + write", ms: ms(t) });

    // ── Report ────────────────────────────────────────────────────────────────
    let total_ms = t_total.elapsed().as_millis();
    let (out_w, out_h) = final_image.dimensions();

    eprintln!();
    eprintln!(
        "  {:<28}  {:>6}",
        "phase", "ms"
    );
    eprintln!("  {}", "-".repeat(38));
    for t in &timings {
        eprintln!("  {:<28}  {:>6}", t.label, t.ms);
    }
    eprintln!("  {}", "-".repeat(38));
    eprintln!("  {:<28}  {:>6}", "total", total_ms);
    eprintln!();
    eprintln!(
        "  input:   {}×{}  raw={}",
        img_w, img_h, is_raw
    );
    eprintln!("  output:  {}×{}  {}", out_w, out_h, args.output.display());

    Ok(())
}

fn run_server() -> Result<(), String> {
    use std::io::Write;

    let state = make_state();
    let context = init_gpu_context_headless(&state)?;
    eprintln!("ready");

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let result = match serde_json::from_str::<Job>(line) {
            Err(e) => JobResult::Err { ok: false, id: None, error: format!("invalid job JSON: {e}") },
            Ok(job) => {
                let id = job.id.clone();
                match process_image(&job.input, &job.preset, &job.output, job.quality, job.resize, &context, &state) {
                    Ok(ms) => JobResult::Ok { ok: true, id, ms },
                    Err(e) => JobResult::Err { ok: false, id, error: e },
                }
            }
        };

        let mut json = serde_json::to_string(&result).unwrap();
        json.push('\n');
        out.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn main() {
    init_logging();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            print_usage();
            std::process::exit(1);
        }
    };
    let result = if args.server {
        run_server()
    } else {
        run(args)
    };
    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
