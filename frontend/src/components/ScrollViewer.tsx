import { useRef, useEffect, useState } from "react";
import type { ScrollViewerProps } from "../interfaces/ScrollViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const ZOOM_STEP = 0.1;

export default function ScrollViewer({ axis, state, onGlobalIndexDelta, volumeKey, seed, setSeed, focusPointChange , isFocusPoint, setIsFocusPoint}: ScrollViewerProps) {
  const [dragging, setDragging] = useState(false);
  const [startDrag, setStartDrag] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [scale, setScale] = useState(0.2);
  const [offset, setOffset] = useState({ x: 0, y: 0});
  const [minZoom, setMinZoom] = useState(0.1);
  const [shiftHeld, setShiftHeld] = useState(false);
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

  //Check if holding shift so we can update cursor style
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Shift") setShiftHeld(true);
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") setShiftHeld(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  function handleMouseDown(e: React.MouseEvent){
    if (e.button !== 2) return; //right click
    setDragging(true);
    setStartDrag({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    e.preventDefault();
  };

  function handleMouseMove(e: React.MouseEvent){
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

  function handleMouseUp() {
    setDragging(false);
  }

  //Setting focus point
  function handleLeftClick(e: React.MouseEvent){
    if (!containerRef.current || !imgRef.current || !e.shiftKey) return;
    const rect = containerRef.current.getBoundingClientRect();

    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const imgX = (cx - offset.x) / scale;
    const imgY = (cy - offset.y) / scale;

    if (imgX < 0 || imgY < 0 || imgX >= imgRef.current.naturalWidth || imgY >= imgRef.current.naturalHeight) 
      return;

    setSeed((prev: { x: number | null; y: number | null; z: number | null }) => {
      const next =
            axis === "x" ? { ...prev, y: Math.round(imgX), z: Math.round(imgY), x: state.index } : 
            axis === "y" ? { ...prev, x: Math.round(imgX), z: Math.round(imgY), y: state.index }
            : { ...prev, x: Math.round(imgX), y: Math.round(imgY), z: state.index };
      
      focusPointChange(next);
      return next;
    });
    setIsFocusPoint(true);
  };

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
            onClick={handleLeftClick}
            style={{
                position: "absolute",
                transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
                transformOrigin: "top left",
                userSelect: "none",
                cursor: shiftHeld ? "default" : dragging ? "grabbing" : "grab"
            }}
            />
          {(() => {
            if (!isFocusPoint) return null;
            if (!imgRef.current) return null;
            if (seed.x == null || seed.y == null || seed.z == null) return null;
    
            let imgX: number | null = null;
            let imgY: number | null = null;

            if (axis === "x") {
              imgX = seed.y ?? null;
              imgY = seed.z ?? null;
            } 
            else if (axis === "y") {
              imgX = seed.x ?? null;
              imgY = seed.z ?? null;
            } 
            else {
              imgX = seed.x ?? null;
              imgY = seed.y ?? null;
            }

            if (imgX == null || imgY == null) return null;

            const screenX = offset.x + imgX * scale;
            const screenY = offset.y + imgY * scale;

            //For transparency the farther away you get from the focus point
            const sliceDistance = Math.abs(state.index - (axis === "x" ? seed.x - ((seed.x - 1) % 4) : axis === "y" ? seed.y - ((seed.y - 1) % 4) : seed.z - ((seed.z - 1) % 4)));
            const MAX_DIST = 20;
            const MIN_ALPHA = 0.05;
            const alpha = sliceDistance > MAX_DIST ? 0 : MIN_ALPHA + (1 - MIN_ALPHA) * (1 - sliceDistance / MAX_DIST);
            if (alpha < MIN_ALPHA) return null;

            return (
              <div
                style={{
                  position: "absolute",
                  left: screenX,
                  top: screenY,
                  transform: "translate(-50%, -50%)",
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  border: `4px solid rgba(0, 255, 255, ${alpha})`, //cyan ring
                  backgroundColor: "transparent",
                  pointerEvents: "none",
                  zIndex: 10
                }}
              />
            );
          })()}
        <div className="overlay" />
      </div>
    </div>
  );
}
