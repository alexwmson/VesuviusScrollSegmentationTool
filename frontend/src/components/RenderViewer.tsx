import type { AxisState } from "../types/AxisState";
import type { RenderViewerProps } from "../interfaces/RenderViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;

export default function RenderViewer({jobId, state, onIndexChange}: RenderViewerProps) {
    const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
        e.preventDefault();

        const delta = Math.sign(e.deltaY);
        const next = Math.min(
            state.max,
            Math.max(state.min, state.index + delta)
        );

        if (next !== state.index) {
            onIndexChange(next);
        }
    };

    return (
        <div className="viewport" onWheel={handleWheel}>
            <div className="image-container">
                <img
                    src={`${API_BASE}/jobs/${jobId}/pngrenders/${state.index}`}
                />
                <div className="overlay" />
            </div>
        </div>
    );
}