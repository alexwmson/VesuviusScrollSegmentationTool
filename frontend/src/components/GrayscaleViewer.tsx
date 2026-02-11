import { useEffect, useRef, useState } from "react";
import type { GrayscaleViewerProps } from "../interfaces/GrayscaleViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const WORLD_SIZE = 10000;
const ZOOM_STEP = 0.1;

export default function GrayscaleViewer({ jobId }: GrayscaleViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);

    const [scale, setScale] = useState(0.2);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const [startDrag, setStartDrag] = useState({ x: 0, y: 0 });
    const [minZoom, setMinZoom] = useState(0.1);
    const [maxZoom, setMaxZoom] = useState(5);
    const [imagePos, setImagePos] = useState({ x: 0, y: 0 });

    // Wheel → zoom
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        const handleWheel = (e: WheelEvent) => {
            e.preventDefault();
            if (!containerRef.current) return;

            const oldScale = scale;
            const zoomDelta = -Math.sign(e.deltaY) * ZOOM_STEP;
            const newScale = Math.min(maxZoom, Math.max(minZoom, oldScale + zoomDelta));
            if (newScale === oldScale) return;

            const rect = containerRef.current.getBoundingClientRect();

            const cursorX = e.clientX - rect.left;
            const cursorY = e.clientY - rect.top;

            const worldX = (cursorX - offset.x) / oldScale;
            const worldY = (cursorY - offset.y) / oldScale;

            let newOffsetX = cursorX - worldX * newScale;
            let newOffsetY = cursorY - worldY * newScale;

            newOffsetX = clampCamera(newOffsetX, rect.width, newScale);
            newOffsetY = clampCamera(newOffsetY, rect.height, newScale);

            setScale(newScale);
            setOffset({ x: newOffsetX, y: newOffsetY });
        };

    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
    }, [scale, offset, minZoom, maxZoom]);

    // Center image on load / resize
    const centerImage = () => {
        const img = imgRef.current;
        const container = containerRef.current;
        if (!img || !container) return;

        const rect = container.getBoundingClientRect();
        const x = (rect.width - img.naturalWidth * scale) / 2;
        const y = (rect.height - img.naturalHeight * scale) / 2;
        setOffset({ x, y });
    };

    const onImageLoad = () => {
        const img = imgRef.current;
        const container = containerRef.current;
        if (!img || !container) return;

        const rect = container.getBoundingClientRect();

        const worldFitZoom = Math.min(rect.width / WORLD_SIZE, rect.height / WORLD_SIZE);
        const imageFitZoom = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
        const maxZ = 8;

        setMinZoom(worldFitZoom);
        setMaxZoom(maxZ);

        setImagePos({x: (WORLD_SIZE - img.naturalWidth) / 2, y: (WORLD_SIZE - img.naturalHeight) / 2});

        const startZoom = Math.min(maxZ, Math.max(worldFitZoom, imageFitZoom));
        setScale(startZoom);

        setOffset({x: rect.width / 2 - (WORLD_SIZE / 2) * startZoom, y: rect.height / 2 - (WORLD_SIZE / 2) * startZoom});
    };

    // Right-click pan
    const handleMouseDown = (e: React.MouseEvent) => {
        if (e.button !== 2) return;
        setDragging(true);
        setStartDrag({ x: e.clientX - offset.x, y: e.clientY - offset.y });
        e.preventDefault();
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!dragging || !containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();

        let x = e.clientX - startDrag.x;
        let y = e.clientY - startDrag.y;

        x = clampCamera(x, rect.width, scale);
        y = clampCamera(y, rect.height, scale);

        setOffset({ x, y });
    };


    const handleMouseUp = () => setDragging(false);

    const clampCamera = (value: number, containerSize: number, scale: number) => {
        const worldPx = WORLD_SIZE * scale;
        const min = containerSize - worldPx;
        return Math.max(min, Math.min(0, value));
    };

    return (
    <div
        ref={containerRef}
        className="viewport"
        onContextMenu={e => e.preventDefault()}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
        overflow: "hidden",
        position: "relative",
        backgroundColor: "rgb(50, 50, 50)"
        }}
    >
        <img
            ref={imgRef}
            src={`${API_BASE}/jobs/${jobId}/grayscale`}
            draggable={false}
            onLoad={onImageLoad}
            style={{
                position: "absolute",
                transform: `translate(${offset.x + imagePos.x * scale}px, ${offset.y + imagePos.y * scale}px) scale(${scale})`,
                transformOrigin: "top left",
                userSelect: "none",
                cursor: dragging ? "grabbing" : "grab"
            }}
        />
    </div>
  );
}
