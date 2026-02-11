import { useEffect, useState } from "react";
import ScrollViewer from "./ScrollViewer";
import GrayscaleViewer from "./GrayscaleViewer";
import RenderViewer from "./RenderViewer";
import type { Axis } from "../types/Axis"
import type { AxisState } from "../types/AxisState";
import type { ImageViewerProps } from "../interfaces/ImageViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
const STEP = 4;
export default function ImageViewer({volumeKey, jobId}: ImageViewerProps){
    const [renderState, setRenderState] = useState<AxisState>({
        index: 1,
        min: 1,
        max: 1
    });

    const [axes, setAxes] = useState<Record<Axis, AxisState>>({
        x: { index: 1, min: 1, max: 1 },
        y: { index: 1, min: 1, max: 1 },
        z: { index: 1, min: 1, max: 1 }
    });
    const [focusCoords, setFocusCoords] = useState(null);

    const applyGlobalDelta = (direction: number) => {
        if (axes.x.max === 1 || axes.y.max === 1 || axes.z.max === 1)
            return;

        setAxes(prev => {
            const next = { ...prev };

            (["x", "y", "z"] as Axis[]).forEach(axis => {
            const a = prev[axis];
            const nextIndex = a.index + direction * STEP;

            next[axis] = {
                ...a,
                index: Math.min(a.max, Math.max(a.min, nextIndex))
            };
            });

            return next;
        });
        };

    useEffect(() => {
        (["x", "y", "z"] as Axis[]).forEach(async axis => {
            const res = await fetch(
            `${API_BASE}/volumes/${volumeKey}/${axis}`
            );
            const data = await res.json();
            setAxes(prev => ({
            ...prev,
            [axis]: {
                index: data.min_index,
                min: data.min_index,
                max: 1 + STEP * (data.max_index - 1)
            }
            
            }));
        });
    }, [volumeKey]);

    useEffect(() => {
        if (!jobId) return;
        const loadRenderInfo = async () => {
            const res = await fetch(`${API_BASE}/jobs/${jobId}/pngrenders`);
            if (!res.ok) return;

            const data = await res.json();

            setRenderState({
                index: data.min_index,
                min: data.min_index,
                max: data.max_index
            });
        };

        loadRenderInfo();
    }, [jobId]);

    return (
        <main>
            <div id="renderdiv">
                <div id="firstrow">
                    <GrayscaleViewer jobId={jobId}/>
                    <ScrollViewer
                        axis="z"
                        volumeKey={volumeKey}
                        state={axes.z}
                        onGlobalIndexDelta={applyGlobalDelta}
                    />
                </div>
                <div id="secondrow">
                    <ScrollViewer
                        axis="x"
                        volumeKey={volumeKey}
                        state={axes.x}
                        onGlobalIndexDelta={applyGlobalDelta}
                    />
                    <ScrollViewer
                        axis="y"
                        volumeKey={volumeKey}
                        state={axes.y}
                        onGlobalIndexDelta={applyGlobalDelta}
                    />
                </div>
            </div>
        </main>
    );
}