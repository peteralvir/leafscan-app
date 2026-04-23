import { useState, useRef, useCallback } from "react";
import * as ort from "onnxruntime-web";

// Point WASM to the CDN so it loads correctly in Vite
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

const CLASSES = [
  "Tomato Bacterial Spot",                    // 0 - Tomato_Bacterial_spot
  "Tomato Early Blight",                      // 1 - Tomato_Early_blight
  "Tomato Late Blight",                       // 2 - Tomato_Late_blight
  "Tomato Leaf Mold",                         // 3 - Tomato_Leaf_Mold
  "Tomato Septoria Leaf Spot",                // 4 - Tomato_Septoria_leaf_spot
  "Tomato Spider Mites (Two-spotted)",        // 5 - Tomato_Spider_mites_Two_spotted_spider_mite
  "Tomato Target Spot",                       // 6 - Tomato__Target_Spot
  "Tomato Yellow Leaf Curl Virus",            // 7 - Tomato__Tomato_YellowLeaf__Curl_Virus
  "Tomato Mosaic Virus",                      // 8 - Tomato__Tomato_mosaic_virus
  "Tomato Healthy",                           // 9 - Tomato_healthy
];

const CLASS_INFO = {
  "Tomato Bacterial Spot":           { color: "#e05a3a", severity: "High" },
  "Tomato Early Blight":             { color: "#c97c2e", severity: "Moderate" },
  "Tomato Late Blight":              { color: "#b03030", severity: "High" },
  "Tomato Leaf Mold":                { color: "#7a9e3a", severity: "Moderate" },
  "Tomato Septoria Leaf Spot":       { color: "#c97c2e", severity: "Moderate" },
  "Tomato Spider Mites (Two-spotted)":{ color: "#a04060", severity: "High" },
  "Tomato Target Spot":              { color: "#c97c2e", severity: "Moderate" },
  "Tomato Yellow Leaf Curl Virus":   { color: "#c9a000", severity: "Moderate" },
  "Tomato Mosaic Virus":             { color: "#7a3090", severity: "High" },
  "Tomato Healthy":                  { color: "#3a9e5a", severity: "None" },
};

const severityColor = { None: "#3a9e5a", Moderate: "#c97c2e", High: "#b03030" };

let tfliteModel = null;

let ortSession = null;

async function loadModel() {
  if (ortSession) return ortSession;
  ortSession = await ort.InferenceSession.create(
    "assets/cbam_fusion_seed123.onnx",
    { executionProviders: ["wasm"] }
  );
  return ortSession;
}

async function classifyLeaf(imageDataUrl) {
  const session = await loadModel();

  // Preprocess image
  const img = new Image();
  img.src = imageDataUrl;
  await new Promise((r) => (img.onload = r));

  const canvas = document.createElement("canvas");
  canvas.width = 224; canvas.height = 224;
  canvas.getContext("2d").drawImage(img, 0, 0, 224, 224);

  const imageData = canvas.getContext("2d").getImageData(0, 0, 224, 224);
  const { data } = imageData;

  // Convert to float32 tensor [1, 3, 224, 224] — ONNX uses channels first
  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const float32 = new Float32Array(1 * 3 * 224 * 224);

  for (let i = 0; i < 224 * 224; i++) {
    float32[0 * 224 * 224 + i] = (data[i * 4 + 0] / 255.0 - mean[0]) / std[0]; // R
    float32[1 * 224 * 224 + i] = (data[i * 4 + 1] / 255.0 - mean[1]) / std[1]; // G
    float32[2 * 224 * 224 + i] = (data[i * 4 + 2] / 255.0 - mean[2]) / std[2]; // B
  }

  const tensor = new ort.Tensor("float32", float32, [1, 3, 224, 224]);
  const feeds  = { input: tensor };
  const result = await session.run(feeds);

  // Get output and apply softmax
  const logits = result.output.data;
  const expArr = Array.from(logits).map(Math.exp);
  const expSum = expArr.reduce((a, b) => a + b, 0);
  const probs  = expArr.map((e) => e / expSum);

  const indexed = probs
    .map((p, i) => ({ class: CLASSES[i], confidence: p }))
    .sort((a, b) => b.confidence - a.confidence);

  return {
    class:      indexed[0].class,
    confidence: parseFloat(indexed[0].confidence.toFixed(4)),
    top3:       indexed.slice(0, 3).map((x) => ({
      class: x.class,
      confidence: parseFloat(x.confidence.toFixed(4)),
    })),
    note: `On-device · Seed 123 · ${(indexed[0].confidence * 100).toFixed(1)}% confidence`,
  };
}

function ConfidenceBar({ value, color }) {
  return (
    <div style={{ background: "#1e2a1e", borderRadius: 6, height: 6, overflow: "hidden", flex: 1 }}>
      <div style={{ height: "100%", width: `${(value * 100).toFixed(1)}%`,
        background: color, borderRadius: 6, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)" }} />
    </div>
  );
}

export default function App() {
  const [imagePreview, setImagePreview]   = useState(null);
  const [imageDataUrl, setImageDataUrl]   = useState(null);
  const [status, setStatus]               = useState("idle");
  const [statusMsg, setStatusMsg]         = useState("");
  const [result, setResult]               = useState(null);
  const [dragOver, setDragOver]           = useState(false);
  const fileInputRef = useRef();

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!["image/jpeg","image/png","image/jpg"].includes(file.type)) {
      setStatus("error");
      setStatusMsg("Invalid format. Please upload JPEG or PNG.");
      return;
    }
    setStatus("uploading");
    setStatusMsg("Uploading image…");
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target.result);
      setImageDataUrl(e.target.result);
      setStatus("idle");
      setStatusMsg("Image ready. Press Check for Plant Disease.");
    };
    reader.onerror = () => { setStatus("error"); setStatusMsg("Failed to read file."); };
    reader.readAsDataURL(file);
  }, []);

  const handleUploadClick = () => { setResult(null); fileInputRef.current?.click(); };
  const handleFileChange  = (e) => { handleFile(e.target.files?.[0]); e.target.value = ""; };
  const handleDrop = (e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files?.[0]); };

  const handleAnalyze = async () => {
    if (!imageDataUrl) { setStatus("error"); setStatusMsg("No image selected."); return; }
    setStatus("analyzing"); setStatusMsg("Analyzing leaf…"); setResult(null);
    try {
      const res = await classifyLeaf(imageDataUrl);
      setResult(res); setStatus("done"); setStatusMsg("Classification complete.");
    } catch (e) {
      console.error("Full error:", e);
      setStatus("error");
      setStatusMsg(`Error: ${e.message}`);
    }
  };

  const resultColor = result ? CLASS_INFO[result.class]?.color || "#4a9e6a" : "#4a9e6a";

  return (
      <div style={{ minHeight:"100vh", background:"#0d1a0d", fontFamily:"'DM Sans','Segoe UI',sans-serif",
        display:"flex", flexDirection:"column", alignItems:"center", padding:"0 0 40px", color:"#e8f0e8",
        width:"100%", boxSizing:"border-box" }}>

      {/* Header */}
      <div style={{ width:"100%", borderBottom:"1px solid #1e331e", padding:"20px 5vw 16px", textAlign:"center", position:"relative", boxSizing:"border-box" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}>
        <img src="assets/leaf-svgrepo-com.svg" alt="LeafScan logo" style={{ width:24, height:24 }} />
        <h1 style={{ margin:0, fontSize:20, fontWeight:700, color:"#c8e6c8" }}>LeafScan</h1>
      </div>
        <p style={{ margin:"4px 0 0", fontSize:12, color:"#5a7a5a", letterSpacing:"0.5px" }}>TOMATO LEAF DISEASE CLASSIFIER</p>
      </div>

      <div style={{ width:"100%", maxWidth:480, padding:"0 5vw", boxSizing:"border-box" }}>

        {/* Image upload area */}
        <div style={{ marginTop:24 }}>
          <div onDrop={handleDrop} onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => !imagePreview && handleUploadClick()}
            style={{ width:"100%", aspectRatio:"4/3", background: dragOver ? "#1a2f1a" : "#111c11",
              border:`2px dashed ${dragOver ? "#4a8a4a" : "#2a3a2a"}`, borderRadius:16,
              display:"flex", alignItems:"center", justifyContent:"center", overflow:"hidden",
              cursor: imagePreview ? "default" : "pointer", position:"relative", transition:"all 0.2s" }}>
            {imagePreview ? (
              <>
                <img src={imagePreview} alt="Leaf" style={{ width:"100%", height:"100%", objectFit:"cover" }} />
                {status === "analyzing" && (
                  <div style={{ position:"absolute", inset:0, background:"rgba(0,0,0,0.65)",
                    display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:12 }}>
                    <div style={{ width:44, height:44, border:"3px solid #2a4a2a", borderTop:"3px solid #6aaa6a",
                      borderRadius:"50%", animation:"spin 0.8s linear infinite" }} />
                    <span style={{ fontSize:13, color:"#9abf9a" }}>Analyzing leaf…</span>
                  </div>
                )}
              </>
            ) : (
              <div style={{ textAlign:"center", color:"#3a5a3a" }}>
                <svg width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg"
                  style={{ marginBottom:8, opacity:0.35 }}>
                  <path d="M36 8C24 8 10 20 10 40C10 54 22 64 36 64C50 64 62 54 62 40C62 20 48 8 36 8Z"
                    fill="#2d4a2d" stroke="#4a7a4a" strokeWidth="1.5"/>
                  <path d="M36 8 C36 8 36 64 36 64" stroke="#6aaa6a" strokeWidth="1.5" strokeLinecap="round"/>
                  <path d="M36 24 C26 21 14 27 11 36" stroke="#6aaa6a" strokeWidth="1.2" strokeLinecap="round"/>
                  <path d="M36 36 C26 33 15 40 12 48" stroke="#6aaa6a" strokeWidth="1.2" strokeLinecap="round"/>
                  <path d="M36 24 C46 21 58 27 61 36" stroke="#6aaa6a" strokeWidth="1.2" strokeLinecap="round"/>
                  <path d="M36 36 C46 33 57 40 60 48" stroke="#6aaa6a" strokeWidth="1.2" strokeLinecap="round"/>
                </svg>
                <p style={{ margin:0, fontSize:13 }}>Tap to upload</p>
                <p style={{ margin:"4px 0 0", fontSize:11, color:"#2a3a2a" }}>JPEG or PNG only</p>
              </div>
            )}
          </div>
          <input ref={fileInputRef} type="file" accept="image/jpeg,image/png"
            onChange={handleFileChange} style={{ display:"none" }} />
        </div>

        {/* Result panel */}
        <div style={{ marginTop:18 }}>
          <p style={{ margin:"0 0 8px", fontSize:12, color:"#4a6a4a", letterSpacing:"0.5px", fontWeight:600 }}>RESULT</p>
          <div style={{ background:"#111c11", border:`1px solid ${result ? resultColor+"55" : "#1e2e1e"}`,
            borderRadius:12, padding:"14px 16px", minHeight:54, display:"flex",
            alignItems:"center", justifyContent:"space-between", transition:"border-color 0.4s" }}>
            {result ? (
              <>
                <div>
                  <p style={{ margin:0, fontSize:16, fontWeight:600, color:resultColor }}>{result.class}</p>
                  <p style={{ margin:"2px 0 0", fontSize:11, color:"#4a6a4a" }}>{(result.confidence*100).toFixed(1)}% confidence</p>
                </div>
                <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:2 }}>
                  <span style={{ fontSize:9, color:"#3a5a3a", letterSpacing:"0.5px" }}>SEVERITY</span>
                  <div style={{ fontSize:11, padding:"4px 10px", borderRadius:6,
                    background: severityColor[CLASS_INFO[result.class]?.severity]+"22",
                    color: severityColor[CLASS_INFO[result.class]?.severity],
                    border:`1px solid ${severityColor[CLASS_INFO[result.class]?.severity]}44`, fontWeight:500 }}>
                    {CLASS_INFO[result.class]?.severity}
                  </div>
                </div>
              </>
            ) : (
              <p style={{ margin:0, fontSize:14, color:"#2a3a2a" }}>
                {status === "analyzing" ? "Classifying…" : "Awaiting classification"}
              </p>
            )}
          </div>
        </div>

        {/* Top 3 */}
        {result?.top3 && (
          <div style={{ marginTop:12, background:"#0d160d", border:"1px solid #1a2a1a",
            borderRadius:12, padding:"12px 14px" }}>
            <p style={{ margin:"0 0 10px", fontSize:11, color:"#3a5a3a", letterSpacing:"0.5px", fontWeight:600 }}>TOP PREDICTIONS</p>
            {result.top3.map((item, i) => (
              <div key={i} style={{ display:"flex", alignItems:"center", gap:10, marginBottom: i<2 ? 8 : 0 }}>
                <span style={{ fontSize:11, color:"#3a5a3a", minWidth:14 }}>{i+1}</span>
                <span style={{ fontSize:12, color: i===0 ? "#c8e6c8" : "#4a6a4a", flex:"0 0 175px",
                  whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis" }}>{item.class}</span>
                <ConfidenceBar value={item.confidence} color={i===0 ? resultColor : "#2a4a2a"} />
                <span style={{ fontSize:11, color:"#4a6a4a", minWidth:36, textAlign:"right" }}>
                  {(item.confidence*100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Buttons */}
        <div style={{ marginTop:20, display:"flex", flexDirection:"column", gap:10 }}>
          <button onClick={handleUploadClick}
            style={{ width:"100%", padding:"14px 20px", background:"#1a2a1a", border:"1px solid #2a3a2a",
              borderRadius:12, color:"#8ab88a", fontSize:14, fontWeight:500, cursor:"pointer" }}>
            ⬆ Upload File
          </button>
          <button onClick={handleAnalyze} disabled={!imageDataUrl || status==="analyzing"}
            style={{ width:"100%", padding:"14px 20px", cursor: !imageDataUrl||status==="analyzing" ? "not-allowed":"pointer",
              background: !imageDataUrl||status==="analyzing" ? "#0d160d":"#1e3a1e",
              border:`1px solid ${!imageDataUrl||status==="analyzing" ? "#1a2a1a":"#3a6a3a"}`,
              borderRadius:12, color: !imageDataUrl||status==="analyzing" ? "#2a3a2a":"#7acc7a",
              fontSize:14, fontWeight:600 }}>
            Check for Plant Disease
          </button>
        </div>

        {/* Status */}
        <div style={{ marginTop:20, display:"flex", alignItems:"center", gap:8, padding:"0 4px" }}>
          <div style={{ width:7, height:7, borderRadius:"50%",
            background: status==="done" ? "#3a9e5a" : status==="error" ? "#b03030" :
              status==="analyzing" ? "#c9a000" : "#2a3a2a",
            boxShadow: status==="analyzing" ? "0 0 6px #c9a000aa" : status==="done" ? "0 0 6px #3a9e5aaa" : "none" }} />
          <span style={{ fontSize:12, color: status==="error" ? "#b03030" : "#3a5a3a" }}>
            {statusMsg || (imageDataUrl ? "Image loaded. Ready to classify." : "No image selected.")}
          </span>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}