import axios from 'axios';
import { Segment, Volume, GenerateSegmentParams } from '../types';

const API_BASE = 'http://localhost:3000/api'; // Adjust to your backend URL

export const getSegments = async (): Promise<Segment[]> => {
  const response = await axios.get(`${API_BASE}/segments`);
  return response.data;
};

export const getVolumes = async (): Promise<Volume[]> => {
  const response = await axios.get(`${API_BASE}/volumes`);
  return response.data;
};

export const getSegmentPng = async (jobid: string, slice: number): Promise<string> => {
  const response = await axios.get(`${API_BASE}/segment/${jobid}/png/${slice}`, { responseType: 'blob' });
  return URL.createObjectURL(response.data);
};

export const getVolumePng = async (id: string, axis: 'x' | 'y' | 'z', slice: number): Promise<string> => {
  const response = await axios.get(`${API_BASE}/volume/${id}/png/${axis}/${slice}`, { responseType: 'blob' });
  return URL.createObjectURL(response.data);
};

export const generateSegment = async (params: GenerateSegmentParams): Promise<void> => {
  await axios.post(`${API_BASE}/generate-segment`, params);
};
