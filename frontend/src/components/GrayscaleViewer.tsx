import type { GrayscaleViewerProps } from "../interfaces/GrayscaleViewerProps"

const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function GrayscaleViewer({jobId}: GrayscaleViewerProps){
    return (
        <img src={`${API_BASE}/jobs/${jobId}`}/>
    )
}