import { useEffect, useState } from "react";
import ScrollViewer from "./ScrollViewer";
import GrayscaleViewer from "./GrayscaleViewer";
import RenderViewer from "./RenderViewer";
import type { Axis } from "../types/Axis"
import type { AxisState } from "../types/AxisState";
import type { ImageViewerProps } from "../interfaces/ImageViewerProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
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

    const handleIndexChange = (axis: Axis, index: number) => {
        setAxes(prev => ({
            ...prev,
            [axis]: {
            ...prev[axis],
            index
            }
        }));
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
                max: data.max_index
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
            { jobId.length > 0 &&
                <div id="grayscalediv">
                    <GrayscaleViewer jobId={jobId}/>
                </div>
            }
            <div id="renderdiv">
                <div id="firstrow">
                    { jobId.length > 0 &&
                        <RenderViewer jobId={jobId} state={renderState} onIndexChange={(index) => setRenderState(prev => ({ ...prev, index }))}/>
                    }
                    <ScrollViewer
                        axis="z"
                        volumeKey={volumeKey}
                        state={axes.z}
                        onIndexChange={handleIndexChange}
                    />
                </div>
                <div id="secondrow">
                    <ScrollViewer
                        axis="x"
                        volumeKey={volumeKey}
                        state={axes.x}
                        onIndexChange={handleIndexChange}
                    />
                    <ScrollViewer
                        axis="y"
                        volumeKey={volumeKey}
                        state={axes.y}
                        onIndexChange={handleIndexChange}
                    />
                </div>
            </div>
        </main>
    );
}