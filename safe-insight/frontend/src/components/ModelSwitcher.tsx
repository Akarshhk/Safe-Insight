import { useEffect, useState } from "react";
import { getModelCatalog, switchModel, BASE_URL } from "../api";

interface ModelEntry {
  id: string;
  tier: string;
  name: string;
  filename: string;
  size_gb: number;
  min_ram_gb: number;
  description: string;
  fits_ram: boolean;
  downloaded?: boolean;
}

interface ModelSwitcherProps {
  onClose: () => void;
  activeModelFile: string | undefined;
}

export default function ModelSwitcher({ onClose, activeModelFile }: ModelSwitcherProps) {
  const [catalog, setCatalog] = useState<ModelEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelCatalog()
      .then((cat) => {
        setCatalog(cat);
        const active = cat.find((m) => m.filename === activeModelFile);
        if (active) setSelectedModel(active.id);
      })
      .catch((err) => setError("Failed to load catalog: " + err.message));
  }, [activeModelFile]);

  const handleAction = async () => {
    if (!selectedModel) return;
    const model = catalog.find(m => m.id === selectedModel);
    if (!model) return;

    setProcessing(true);
    setError(null);

    if (model.downloaded) {
      // Hot swap immediately
      setStatus("Switching model...");
      try {
        await switchModel(model.filename);
      } catch (err: any) {
        setError(err.message || String(err));
        setProcessing(false);
        return;
      }
      onClose();
    } else {
      // Download then switch
      setProgress(0);
      setStatus("Connecting...");
  
      const source = new EventSource(`${BASE_URL}/setup/download/${selectedModel}`);
      
      source.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
            setError(data.error);
            setProcessing(false);
            source.close();
            return;
          }
          
          if (data.progress !== undefined) setProgress(data.progress);
          if (data.status) setStatus(data.status);
  
          if (data.status === "Complete" && data.progress === 100) {
            source.close();
            // Call switchModel to trigger reload_llm in backend
            switchModel(model.filename).then(() => {
                onClose();
            }).catch(err => {
                setError(err.message || String(err));
                setProcessing(false);
            });
          }
        } catch (err) {
          console.error("Parse error", err);
        }
      };
  
      source.onerror = (err) => {
        setError("Connection lost. You can try again to resume.");
        setProcessing(false);
        source.close();
      };
    }
  };

  return (
    <div className="modal-overlay" style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="setup-wizard__card" style={{ maxWidth: '800px', width: '100%', maxHeight: '90vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2>Model Settings</h2>
          <button className="btn btn--small" onClick={onClose} disabled={processing}>Close</button>
        </div>

        {error && <div className="status status--error" style={{ marginBottom: '20px' }}>{error}</div>}

        <div className="catalog-grid">
          {catalog.map((model) => (
            <div 
              key={model.id} 
              className={`catalog-item ${selectedModel === model.id ? "selected" : ""} ${!model.fits_ram ? "warning" : ""}`}
              onClick={() => !processing && setSelectedModel(model.id)}
            >
              <h3>{model.name} {model.filename === activeModelFile && <span className="badge">Active</span>}</h3>
              <p className="size">{model.size_gb} GB</p>
              <p className="desc">{model.description}</p>
              {!model.fits_ram && (
                <p className="ram-warning">⚠️ This model exceeds your available RAM.</p>
              )}
              {model.downloaded && model.filename !== activeModelFile && (
                <p className="status status--ok">Downloaded</p>
              )}
            </div>
          ))}
        </div>

        <div style={{ marginTop: '20px' }}>
          {processing && progress > 0 ? (
            <div className="download-progress">
              <div className="bar-bg">
                <div className="bar-fg" style={{ width: `${progress}%` }}></div>
              </div>
              <p>{status} {progress > 0 ? `${progress}%` : ""}</p>
            </div>
          ) : processing ? (
            <p className="status status--busy">{status}</p>
          ) : (
            <button 
              className="btn btn--primary" 
              disabled={!selectedModel || catalog.length === 0} 
              onClick={handleAction}
            >
              {catalog.find(m => m.id === selectedModel)?.downloaded 
                ? "Switch Model" 
                : "Download and Switch"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
