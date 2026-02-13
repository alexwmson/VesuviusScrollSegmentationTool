import React, { useState } from 'react';
import Header from "./components/Header.tsx";
import ImageViewer from './components/ImageViewer.tsx';
import Form from "./components/Form.tsx";
import Directory from './components/Directory.tsx';
import type { SegmentMeta } from './interfaces/MetaProps.tsx';
import Meta from './components/Meta.tsx';

function App() {
  const [curVolume, setCurVolume] = useState("center_scroll1");
  const [curSegment, setCurSegment] = useState("f7be45d4-b6f2-4a12-9921-7cc145407251");
  const [segments, setSegments] = useState<[string, number][]>([]);
  const [seed, setSeed] = useState<{ x: number | null; y: number | null; z: number | null }>({x: null, y: null, z: null});
  const [isFocusPoint, setIsFocusPoint] = useState<boolean>(false);
  const [meta, setMeta] = useState<SegmentMeta | null>(null);

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
