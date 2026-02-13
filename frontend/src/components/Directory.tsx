import type DirectoryProps from "../interfaces/DirectoryProps";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function Directory({segments, setCurSegment, curSegment, setCurVolume, curVolume, setMeta} : DirectoryProps){
    const paddedSegments = [
    ...segments,
    ...Array(Math.max(0, 10 - segments.length)).fill([null, 0])
    ];

    async function handleSegmentChange(uuid: string){
       setCurSegment(uuid);
        try {
            const res = await (await fetch(`${API_BASE}/jobs/${uuid}/meta`)).json();
            console.log(res);
            setMeta({
                starting_points: [res.starting_point[0], res.starting_point[1], res.starting_point[2]],
                voxels_explored: res.voxels_explored,
                pixels_on_grayscale: res.pixels_on_grayscale,
                created_on: res.created_on
            })
        }
        catch(e){
            console.error("Failed to get meta", e);
        }

    }

    const segmentElements = paddedSegments.map((segment, index) => (
        segment[0] ? (
        <li key={segment[0]} onClick={() => segment[1] !== 0 ? handleSegmentChange(segment[0]) : function(){}} className={(segment[0] === curSegment ? "active" : "") + " " + (segment[1] === 0 ? "notdone" : "")}>
            segment_{segment[0]}
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