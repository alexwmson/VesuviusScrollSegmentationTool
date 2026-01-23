import React, { useState } from 'react';
import Header from "./components/Header.tsx";

function App() {
  const [curVolume, setVolume] = useState("center_scroll1");
  const [posX, setX] = useState(1);
  const [posY, setY] = useState(1);
  const [posZ, setZ] = useState(1);

  return (
    <>
      <Header></Header>
    </>
  );
}

export default App;
