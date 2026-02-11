import { useRef, useEffect, useState } from "react";
import type { ScrollViewerProps } from "../interfaces/ScrollViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const ZOOM_STEP = 0.1;

export default function ScrollViewer({ axis, state, onGlobalIndexDelta, volumeKey }: ScrollViewerProps) {
  const [dragging, setDragging] = useState(false);
  const [startDrag, setStartDrag] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState(0.2);
  const [offset, setOffset] = useState({ x: 0, y: 0});
  const [minZoom, setMinZoom] = useState(0.1);
  const maxZoom = 5;

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (!containerRef.current || !imgRef.current) return;

      if (e.shiftKey) {
        onGlobalIndexDelta(Math.sign(e.deltaY));
        return;
      }

      const container = containerRef.current;
      const img = imgRef.current;
      const oldScale = scale;
      const zoomDelta = -Math.sign(e.deltaY) * ZOOM_STEP;
      const newScale = Math.min(maxZoom, Math.max(minZoom, oldScale + zoomDelta));

      const rect = container.getBoundingClientRect();
      const cursorX = e.clientX - rect.left;
      const cursorY = e.clientY - rect.top;

      let offsetX = cursorX - ((cursorX - offset.x) * newScale / oldScale);
      let offsetY = cursorY - ((cursorY - offset.y) * newScale / oldScale);

      const imgWidth = img.naturalWidth * newScale;
      const imgHeight = img.naturalHeight * newScale;

      const minX = Math.min(0, rect.width - imgWidth);
      const minY = Math.min(0, rect.height - imgHeight);

      offsetX = Math.max(minX, Math.min(0, offsetX));
      offsetY = Math.max(minY, Math.min(0, offsetY));

      setScale(newScale);
      setOffset({ x: offsetX, y: offsetY });
    };

    el.addEventListener("wheel", handleWheel, { passive: false });
    return () => el.removeEventListener("wheel", handleWheel);
  }, [scale, offset, state, onGlobalIndexDelta, axis]);

  useEffect(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (!img || !container) return;

    const containerRect = container.getBoundingClientRect();
    const imgWidth = img.naturalWidth * scale;
    const imgHeight = img.naturalHeight * scale;

    setOffset(prev => {
      let x = prev.x;
      let y = prev.y;

      if (imgWidth <= containerRect.width) x = (containerRect.width - imgWidth) / 2;
      else x = Math.max(containerRect.width - imgWidth, Math.min(0, x));

      if (imgHeight <= containerRect.height) y = (containerRect.height - imgHeight) / 2;
      else y = Math.max(containerRect.height - imgHeight, Math.min(0, y));

      return { x, y };
    });
  }, [state.index, scale]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 2) return; //right click
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

    let x = e.clientX - startDrag.x;
    let y = e.clientY - startDrag.y;

    if (imgWidth <= containerRect.width) x = (containerRect.width - imgWidth) / 2;
    else x = Math.max(containerRect.width - imgWidth, Math.min(0, x));

    if (imgHeight <= containerRect.height) y = (containerRect.height - imgHeight) / 2;
    else y = Math.max(containerRect.height - imgHeight, Math.min(0, y));

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
            onLoad={() => {
                const img = imgRef.current;
                const container = containerRef.current;
                if (!img || !container) return;
                
                const rect = container.getBoundingClientRect();
                let zoom = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
                setMinZoom(zoom);
                setScale(zoom)

                const x = (rect.width - img.naturalWidth * scale) / 2;
                const y = (rect.height - img.naturalHeight * scale) / 2;
                setOffset({ x, y });
            }}
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
