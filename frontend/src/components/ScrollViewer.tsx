import { useRef, useEffect, useState } from "react";
import type { ScrollViewerProps } from "../interfaces/ScrollViewerProps";

const STEP = 4;
const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function ScrollViewer({axis, state, onIndexChange, volumeKey}: ScrollViewerProps){
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const [startDrag, setStartDrag] = useState({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
            const el = containerRef.current;
            if (!el) return;
    
            const handleWheel = (e: WheelEvent) => {
                e.preventDefault();

                const delta = Math.sign(e.deltaY) * STEP;
                const next = Math.min(state.max, Math.max(state.min, state.index + delta));

                if (next !== state.index) {
                onIndexChange(axis, next);
                }
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
        setOffset({ x: e.clientX - startDrag.x, y: e.clientY - startDrag.y });
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
                    src={`${API_BASE}/volumes/${volumeKey}/${axis}/${state.index}`}
                    style={{
                        position: "absolute",
                        top: offset.y,
                        left: offset.x,
                        userSelect: "none",
                        cursor: dragging ? "grabbing" : "grab"
                    }}
                    draggable={false}
                />
                <div className="overlay" />
            </div>
        </div>
    );
}