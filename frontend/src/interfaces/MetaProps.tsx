export interface SegmentMeta {
    starting_points: [number, number, number];
    voxels_explored: number;
    pixels_on_grayscale: number;
    created_on: string;
}

export interface MetaProps{
    meta: SegmentMeta | null;
    uuid: string;
}