import type DirectoryProps from "../interfaces/DirectoryProps";

export default function Directory({segments, setCurSegment, curSegment, setCurVolume, curVolume} : DirectoryProps){
    const paddedSegments = [
    ...segments,
    ...Array(Math.max(0, 10 - segments.length)).fill(null)
    ];

    const segmentElements = paddedSegments.map((segment, index) => (
        segment ? (
        <li key={segment} onClick={() => setCurSegment(segment)} className={segment === curSegment ? "active" : ""}>
            {segment}
        </li>
        ) : (
            <li key={`empty-${index}`} />
        )
    ));
    

    return (
        <div className="directory">
            <select name="volumeDropdown" value={curVolume} onChange={(event: React.ChangeEvent<HTMLSelectElement>) => setCurVolume(event.target.value)}>
                <option id="center_scroll1">center_scroll1</option>
                <option id="leftedge_scroll1">leftedge_scroll1</option>
            </select>
            <ul>
                {segmentElements}
            </ul>
        </div>
    );
}