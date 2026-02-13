export interface ImageViewerProps {
    volumeKey: string;
    jobId: string;
    seed: { x: number | null; y: number | null; z: number | null };
    setSeed: Function;
}