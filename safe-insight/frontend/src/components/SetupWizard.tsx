import { useEffect, useState } from "react";
import { getModelCatalog, BASE_URL } from "../api";

interface ModelEntry {
  id: string;
  tier: string;
  name: string;
  size_gb: number;
  min_ram_gb: number;
  description: string;
  fits_ram: boolean;
  downloaded?: boolean;
}

interface SetupWizardProps {
  onComplete: () => void;
}

export default function SetupWizard({ onComplete }: SetupWizardProps) {
  const [catalog, setCatalog] = useState<ModelEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelCatalog()
      .then((cat) => {
        setCatalog(cat);
        // auto-select balanced
        const balanced = cat.find((m) => m.tier === "balanced");
        if (balanced) setSelectedModel(balanced.id);
      })
      .catch((err) => setError("Failed to load catalog: " + err.message));
  }, []);

  const startDownload = () => {
    if (!selectedModel) return;
    setDownloading(true);
    setError(null);
    setProgress(0);
    setStatus("Connecting...");

    const source = new EventSource(`${BASE_URL}/setup/download/${selectedModel}`);
    
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setError(data.error);
          setDownloading(false);
          source.close();
          return;
        }
        
        if (data.progress !== undefined) setProgress(data.progress);
        if (data.status) setStatus(data.status);

        if (data.status === "Complete" && data.progress === 100) {
          source.close();
          onComplete(); // Model downloaded and set as active, exit wizard
        }
      } catch (err) {
        console.error("Parse error", err);
      }
    };

    source.onerror = () => {
      setError("Connection lost. You can try again to resume.");
      setDownloading(false);
      source.close();
    };
  };

  return (
    <div className="setup-wizard">
      <div className="setup-wizard__card">
        <h1>Welcome to Safe Insight</h1>
        <p className="muted">
          To get started, we need to download the local language model. 
          This is a one-time process and the only time Safe Insight requires the internet.
        </p>

        {error && <div className="status status--error">{error}</div>}

        <div className="catalog-grid">
          {catalog.map((model) => (
            <div 
              key={model.id} 
              className={`catalog-item ${selectedModel === model.id ? "selected" : ""} ${!model.fits_ram ? "warning" : ""}`}
              onClick={() => !downloading && setSelectedModel(model.id)}
            >
              <h3>{model.name} {model.tier === "balanced" && <span className="badge">Recommended</span>}</h3>
              <p className="size">{model.size_gb} GB</p>
              <p className="desc">{model.description}</p>
              {!model.fits_ram && (
                <p className="ram-warning">⚠️ This model exceeds your available RAM.</p>
              )}
              {model.downloaded && (
                <p className="status status--ok">Already downloaded</p>
              )}
            </div>
          ))}
        </div>

        {downloading ? (
          <div className="download-progress">
            <div className="bar-bg">
              <div className="bar-fg" style={{ width: `${progress}%` }}></div>
            </div>
            <p>{status} {progress > 0 ? `${progress}%` : ""}</p>
          </div>
        ) : (
          <button 
            className="btn btn--primary" 
            disabled={!selectedModel || catalog.length === 0} 
            onClick={startDownload}
          >
            {catalog.find(m => m.id === selectedModel)?.downloaded ? "Start (Already Downloaded)" : "Download and Start"}
          </button>
        )}
      </div>
    </div>
  );
}
