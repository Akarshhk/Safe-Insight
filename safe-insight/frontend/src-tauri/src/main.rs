// Safe Insight - Tauri 2 desktop shell.
//
// The shell has exactly two jobs:
//
//   1. Start the Python FastAPI backend as a child process on launch, and make
//      sure it dies when the window closes. An orphaned backend holding the
//      FAISS index open is the single most annoying dev-loop failure, so the
//      teardown path is handled explicitly rather than left to the OS.
//   2. Host the React UI in the system webview.
//
// It does NOT proxy API calls: the webview talks to 127.0.0.1:8765 directly with
// `fetch`, which keeps the Rust layer thin and the pipeline easy to debug with
// curl or the FastAPI docs page at http://127.0.0.1:8765/docs.
//
// v1 spawns the interpreter directly (the brief allows dev-mode spawn). Bundling
// the backend as a PyInstaller sidecar is the stretch goal - see README,
// "Packaging (stretch goal)".

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{Manager, RunEvent, State};

/// Handle to the spawned Python process, stored in Tauri's managed state so the
/// exit handler can reach it.
struct BackendProcess(Mutex<Option<Child>>);

impl BackendProcess {
    /// Terminate the backend if it is still running. Safe to call twice.
    fn shutdown(&self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(mut child) = guard.take() {
                // SIGKILL-equivalent. The backend keeps no unflushed state:
                // the FAISS index and audit DB are written synchronously on
                // every mutation, so an abrupt stop loses nothing.
                let _ = child.kill();
                let _ = child.wait();
                println!("[safe-insight] backend process stopped");
            }
        }
    }
}

/// Locate `backend/` relative to wherever the shell is running from.
///
/// `tauri dev` runs with the current directory set to `src-tauri/`, but a built
/// binary runs from `target/<profile>/`. Rather than guess, walk up from both
/// the current directory and the executable's directory looking for the marker
/// file `backend/app/main.py`.
fn find_backend_dir() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
        }
    }

    for root in roots {
        let mut current: Option<&Path> = Some(root.as_path());
        // Six levels is enough for target/debug/ inside a workspace.
        for _ in 0..6 {
            let candidate = match current {
                Some(dir) => dir.join("backend"),
                None => break,
            };
            if candidate.join("app").join("main.py").is_file() {
                return Some(candidate);
            }
            current = current.and_then(Path::parent);
        }
    }
    None
}

/// Pick the Python interpreter to run.
///
/// A virtualenv inside `backend/` wins, because that is where the user was told
/// to install the dependencies. Otherwise fall back to whatever `python` (or
/// `python3`) is on PATH.
fn find_python(backend_dir: &Path) -> PathBuf {
    let venv_candidates = if cfg!(windows) {
        vec![
            backend_dir.join(".venv").join("Scripts").join("python.exe"),
            backend_dir.join("venv").join("Scripts").join("python.exe"),
        ]
    } else {
        vec![
            backend_dir.join(".venv").join("bin").join("python"),
            backend_dir.join("venv").join("bin").join("python"),
        ]
    };

    for candidate in venv_candidates {
        if candidate.is_file() {
            return candidate;
        }
    }

    // No venv found. `python` is correct on Windows; `python3` elsewhere.
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

/// Spawn `python -m app.main` with `backend/` as the working directory.
///
/// stdout/stderr are inherited so backend logs appear in the same terminal as
/// the Tauri output - which is what you want while developing.
fn spawn_backend() -> Result<Child, String> {
    let backend_dir = find_backend_dir().ok_or_else(|| {
        "Could not locate the backend/ directory (looked for backend/app/main.py). \
         Start the backend manually with `python -m app.main`."
            .to_string()
    })?;

    let python = find_python(&backend_dir);
    println!(
        "[safe-insight] starting backend: {} -m app.main (cwd {})",
        python.display(),
        backend_dir.display()
    );

    let mut command = Command::new(&python);
    command
        .arg("-m")
        .arg("app.main")
        .current_dir(&backend_dir)
        // Unbuffered stdout, so Python log lines appear immediately rather than
        // in 4 KB bursts.
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    // Keep the console window hidden on Windows release builds.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        if !cfg!(debug_assertions) {
            command.creation_flags(CREATE_NO_WINDOW);
        }
    }

    command.spawn().map_err(|error| {
        format!(
            "Failed to launch {}: {error}. Is Python installed and on PATH, or is \
             there a virtualenv at backend/.venv?",
            python.display()
        )
    })
}

/// Exposed to the UI so it can show why the backend is missing, if it is.
#[tauri::command]
fn backend_status(state: State<'_, BackendProcess>) -> serde_json::Value {
    let running = state
        .0
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(false);
    serde_json::json!({ "spawned_by_shell": running, "url": "http://127.0.0.1:8765" })
}

fn main() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![backend_status])
        .setup(|app| {
            match spawn_backend() {
                Ok(child) => {
                    let state: State<'_, BackendProcess> = app.state();
                    *state.0.lock().unwrap() = Some(child);
                }
                Err(error) => {
                    // Not fatal: the developer may already be running the
                    // backend in another terminal. The UI polls /health and
                    // shows its own message if nothing answers.
                    eprintln!("[safe-insight] {error}");
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Safe Insight application")
        .run(|app_handle, event| {
            // Kill the backend on every path out of the app, including the
            // window's close button. `RunEvent` is #[non_exhaustive], hence the
            // catch-all arm.
            match event {
                RunEvent::ExitRequested { .. } | RunEvent::Exit => {
                    let state: State<'_, BackendProcess> = app_handle.state();
                    state.shutdown();
                }
                _ => {}
            }
        });
}
