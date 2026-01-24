import { useState } from "react";
import type { ScrollViewerProps } from "../interfaces/ScrollViewerProps";

const STEP = 4;
const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function ScrollViewer({axis, state, onIndexChange, volumeKey}: ScrollViewerProps){
    const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
        e.preventDefault();

        const delta = Math.sign(e.deltaY) * STEP;
        const next = Math.min(state.max, Math.max(state.min, state.index + delta));

        if (next !== state.index) {
        onIndexChange(axis, next);
        }
    };

    return (
        <div className="viewport" onWheel={handleWheel}>
            <div className="image-container">
                <img src={`${API_BASE}/volumes/${volumeKey}/${axis}/${state.index}`}/>
                <div className="overlay" />
            </div>
        </div>
    );
}