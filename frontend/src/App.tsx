import React, { useState } from 'react';
import Header from "./components/Header.tsx";
import ImageViewer from './components/ImageViewer.tsx';
import Form from "./components/Form.tsx";
import Directory from './components/Directory.tsx';
import type { SegmentMeta } from './interfaces/MetaProps.tsx';
import Meta from './components/Meta.tsx';

function App() {
  const [curVolume, setCurVolume] = useState("center_scroll1");
  const [curSegment, setCurSegment] = useState("example1");
  const [segments, setSegments] = useState<[string, number][]>([["example1", 1]]);
  const [seed, setSeed] = useState<{ x: number | null; y: number | null; z: number | null }>({x: null, y: null, z: null});
  const [isFocusPoint, setIsFocusPoint] = useState<boolean>(false);
  const [meta, setMeta] = useState<SegmentMeta | null>({
  "starting_points": [311, 87, 596],
  "voxels_explored": 250000,
  "pixels_on_grayscale": 106721,
  "created_on": "20260217220033927"
});

  return (
    <>
      <Header></Header>
      <div className="bodyDiv">
        <div className="optionsDiv">
          <div className="segmentDiv">
            <Directory segments={segments} setCurSegment={setCurSegment} curSegment={curSegment} setCurVolume={setCurVolume} curVolume={curVolume} setMeta={setMeta}/>
            <Meta meta={meta} uuid={curSegment}/>
          </div>
          <Form volume={curVolume} setSegments={setSegments} seed={seed} setSeed={setSeed} isFocusPoint={isFocusPoint} setIsFocusPoint={setIsFocusPoint}/>
        </div>
        <ImageViewer volumeKey={curVolume} jobId={curSegment} seed={seed} setSeed={setSeed} isFocusPoint={isFocusPoint} setIsFocusPoint={setIsFocusPoint}></ImageViewer>
      </div>
    </>
  );
}

export default App;
