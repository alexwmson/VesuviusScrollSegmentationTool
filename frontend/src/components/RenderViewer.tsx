import { useRef, useEffect, useState } from "react";
import type { AxisState } from "../types/AxisState";
import type { RenderViewerProps } from "../interfaces/RenderViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 5;
const ZOOM_STEP = 0.1;

export default function RenderViewer({jobId, state, onIndexChange}: RenderViewerProps) {
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const [startDrag, setStartDrag] = useState({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const [scale, setScale] = useState(1);

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        const handleWheel = (e: WheelEvent) => {
            e.preventDefault();

            if (e.shiftKey) { //if shift + scroll
                const delta = Math.sign(e.deltaY);
                const next = Math.min(
                    state.max,
                    Math.max(state.min, state.index + delta)
                );
                if (next !== state.index) onIndexChange(next);
                return;
            }

            // else zoom
            const zoomDelta = -Math.sign(e.deltaY) * ZOOM_STEP;
            setScale(prev =>
                Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev + zoomDelta))
            );
        };

        el.addEventListener("wheel", handleWheel, { passive: false });

        return () => {
            el.removeEventListener("wheel", handleWheel);
        };
    }, [state, onIndexChange]);

    const handleMouseDown = (e: React.MouseEvent) => {
        if (e.button !== 2) return; // right clicks
        setDragging(true);
        setStartDrag({ x: e.clientX - offset.x, y: e.clientY - offset.y });
        e.preventDefault();
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!dragging) return;

        const container = containerRef.current;
        const img = imgRef.current;
        if (!container || !img) return;

        const containerRect = container.getBoundingClientRect();
        const imgRect = img.getBoundingClientRect();

        let x = e.clientX - startDrag.x;
        let y = e.clientY - startDrag.y;

        const minX = containerRect.width - imgRect.width * scale;
        const maxX = 0;

        const minY = containerRect.height - imgRect.height * scale;
        const maxY = 0;
        
        if (imgRect.width <= containerRect.width) {
            x = (containerRect.width - imgRect.width) / 2;
        } else {
            x = Math.max(minX, Math.min(maxX, x));
        }

        if (imgRect.height <= containerRect.height) {
            y = (containerRect.height - imgRect.height) / 2;
        } else {
            y = Math.max(minY, Math.min(maxY, y));
        }

        setOffset({ x, y });
    };

    const handleMouseUp = () => setDragging(false);

    return (
        <div
            className="viewport"
            ref={containerRef}
            onContextMenu={(e) => e.preventDefault()} // stop the stupid right click menu
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ overflow: "hidden", position: "relative" }}
        >
            <div className="image-container">
                <img
                    ref={imgRef}
                    src={`${API_BASE}/jobs/${jobId}/pngrenders/${state.index}`}
                    draggable={false}
                    style={{
                        position: "absolute",
                        transform: `scale(${scale}) translate(${offset.x}px, ${offset.y}px)`,
                        transformOrigin: "top left",
                        userSelect: "none",
                        cursor: dragging ? "grabbing" : "grab"
                    }}
                />
                <div className="overlay" />
            </div>
        </div>
    );
}