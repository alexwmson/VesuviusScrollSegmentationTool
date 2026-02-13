export default interface DirectoryProps{
    segments: [string, number][];
    setCurSegment: Function;
    curSegment: string;
    setCurVolume: Function;
    curVolume: string;
    setMeta: Function;
}