import type {Axis} from "../types/Axis";
import type { AxisState } from "../types/AxisState";

export interface ScrollViewerProps {
  axis: Axis;
  state: AxisState;
  onGlobalIndexDelta: (delta: number) => void;
  volumeKey: string;
}