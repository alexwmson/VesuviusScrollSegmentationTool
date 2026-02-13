export interface FormProps{
    volume: string;
    setSegments: Function;
    seed: { x: number | null; y: number | null; z: number | null };
    setSeed: Function;
    isFocusPoint: boolean;
    setIsFocusPoint: Function;
}