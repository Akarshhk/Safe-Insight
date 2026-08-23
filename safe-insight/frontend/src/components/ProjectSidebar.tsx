import { useEffect, useState } from "react";
import { listProjects, createProject, deleteProject } from "../api";

interface Project {
  id: string;
  name: string;
  created_at: string;
  doc_count: number;
}

interface ProjectSidebarProps {
  activeProjectId: string | null;
  onProjectSelect: (id: string | null) => void;
}

export default function ProjectSidebar({ activeProjectId, onProjectSelect }: ProjectSidebarProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchProjects = async () => {
    setLoading(true);
    try {
      const data = await listProjects();
      setProjects(data);
      if (data.length > 0 && !activeProjectId) {
        onProjectSelect(data[0].id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleNewProject = async () => {
    const name = prompt("Enter project name:", "New Project");
    if (!name) return;
    try {
      const newProj = await createProject(name);
      await fetchProjects();
      onProjectSelect(newProj.id);
    } catch (err) {
      console.error(err);
      alert("Failed to create project");
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!confirm("Delete this project and all its documents/history?")) return;
    try {
      await deleteProject(id);
      if (activeProjectId === id) {
        onProjectSelect(null);
      }
      await fetchProjects();
    } catch (err) {
      console.error(err);
      alert("Failed to delete project");
    }
  };

  return (
    <div className="project-sidebar">
      <div className="sidebar-header">
        <h2>Projects</h2>
        <button className="btn btn--small" onClick={handleNewProject}>+ New</button>
      </div>
      {loading && projects.length === 0 ? (
        <p className="muted" style={{ padding: "1rem" }}>Loading...</p>
      ) : (
        <ul className="project-list">
          {projects.map((p) => (
            <li 
              key={p.id} 
              className={p.id === activeProjectId ? "active" : ""}
              onClick={() => onProjectSelect(p.id)}
            >
              <div className="project-info">
                <strong>{p.name}</strong>
                <span className="muted small">{p.doc_count} docs</span>
              </div>
              <button 
                className="btn btn--icon delete-btn" 
                onClick={(e) => handleDelete(e, p.id)}
                title="Delete project"
              >
                ✕
              </button>
            </li>
          ))}
          {projects.length === 0 && (
            <li className="muted" style={{ padding: "1rem" }}>No projects yet.</li>
          )}
        </ul>
      )}
    </div>
  );
}
