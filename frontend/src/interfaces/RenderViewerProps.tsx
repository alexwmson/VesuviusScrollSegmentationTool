import type { AxisState } from "../types/AxisState";

export interface RenderViewerProps{
    jobId: string;
    state: AxisState;
    onIndexChange: (index: number) => void;
}