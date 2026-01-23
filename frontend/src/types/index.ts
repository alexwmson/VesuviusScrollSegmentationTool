export interface Segment {
  jobid: string;
  // Add other properties as needed, e.g., name, createdAt
}

export interface Volume {
  id: string;
  name: string;
  // Add other properties as needed
}

export interface GenerateSegmentParams {
  volumeId: string;
  x: number;
  y: number;
  z: number;
  globalThreshold: number;
  allowedDifference: number;
  minSize: number;
  maxSize: number;
}
