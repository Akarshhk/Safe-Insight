// Tauri's build script. Generates the context, capability schemas and (on
// Windows) the resource file from tauri.conf.json. Nothing project-specific
// belongs here.
fn main() {
    tauri_build::build()
}
