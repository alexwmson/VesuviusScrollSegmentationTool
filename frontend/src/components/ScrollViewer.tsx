import { useRef, useEffect, useState } from "react";
import type { ScrollViewerProps } from "../interfaces/ScrollViewerProps";

const STEP = 4;
const API_BASE = import.meta.env.VITE_API_BASE_URL;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 5;
const ZOOM_STEP = 0.1;

export default function ScrollViewer({axis, state, onIndexChange, volumeKey}: ScrollViewerProps){
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

        // if shift + scroll
        if (e.shiftKey) {
            const delta = Math.sign(e.deltaY) * STEP;
            const next = Math.min(
                state.max,
                Math.max(state.min, state.index + delta)
            );

            if (next !== state.index) {
                onIndexChange(axis, next);
            }
            return;
        }

        // else zoom normally
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
        if (e.button !== 2) return; // right-click
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

        const imgWidth = img.naturalWidth * scale;
        const imgHeight = img.naturalHeight * scale;

        const viewWidth = containerRect.width;
        const viewHeight = containerRect.height;

        let x = e.clientX - startDrag.x;
        let y = e.clientY - startDrag.y;

        const minX = Math.min(0, viewWidth - imgWidth);
        const maxX = 0;
        x = Math.max(minX, Math.min(maxX, x));

        const minY = Math.min(0, viewHeight - imgHeight);
        const maxY = 0;
        y = Math.max(minY, Math.min(maxY, y));

        setOffset({ x, y });
    };

    const handleMouseUp = () => setDragging(false);

    return (
        <div
            className="viewport"
            ref={containerRef}
            onContextMenu={(e) => e.preventDefault()}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ overflow: "hidden", position: "relative" }}
        >
            <div className="image-container">
                <img
                    ref={imgRef}
                    src={`${API_BASE}/volumes/${volumeKey}/${axis}/${state.index}`}
                    draggable={false}
                    style={{
                        position: "absolute",
                        transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
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