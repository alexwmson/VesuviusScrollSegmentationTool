import type {Axis} from "../types/Axis";
import type { AxisState } from "../types/AxisState";

export interface ScrollViewerProps {
  axis: Axis;
  state: AxisState;
  onGlobalIndexDelta: Function;
  volumeKey: string;
  seed: { x: number | null; y: number | null; z: number | null };
  setSeed: Function;
  focusPointChange: Function;
  isFocusPoint: boolean;
  setIsFocusPoint: Function;
}