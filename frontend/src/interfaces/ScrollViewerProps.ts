import type {Axis} from "../types/Axis";
import type { AxisState } from "../types/AxisState";

export interface ScrollViewerProps {
  axis: Axis;
  state: AxisState;
  onIndexChange: (axis: Axis, index: number) => void;
  volumeKey: string;
}