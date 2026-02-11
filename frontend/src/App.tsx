import React, { useState } from 'react';
import Header from "./components/Header.tsx";
import ImageViewer from './components/ImageViewer.tsx';
import Form from "./components/Form.tsx";
import Directory from './components/Directory.tsx';

function App() {
  const [curVolume, setCurVolume] = useState("center_scroll1");
  const [curSegment, setCurSegment] = useState("f7be45d4-b6f2-4a12-9921-7cc145407251");
  const [segments, setSegments] = useState<[string, number][]>([]);

  return (
    <>
      <Header></Header>
      <div className="bodyDiv">
        <Directory segments={segments} setCurSegment={setCurSegment} curSegment={curSegment} setCurVolume={setCurVolume} curVolume={curVolume}/>
        <ImageViewer volumeKey={curVolume} jobId={curSegment}></ImageViewer>
        <Form volume={curVolume} setSegments={setSegments}/>
      </div>
    </>
  );
}

export default App;
