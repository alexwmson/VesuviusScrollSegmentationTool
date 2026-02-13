import React, { useState } from "react";
import type { FormProps } from "../interfaces/FormProps"
import type { FormStateProp } from "../interfaces/FormStateProp";
import type { ParamsProp } from "../interfaces/FormStateProp";

const API_BASE = import.meta.env.VITE_API_BASE_URL;
export default function Form({volume, setSegments, seed, setSeed, isFocusPoint, setIsFocusPoint} : FormProps){
    const { x, y, z } = seed;
    const [params, setParams] = useState<ParamsProp>({
        globalThreshold: 112,
        allowedDifference: 0.95,
        patienceMax: 5,
        minSize: 50000,
        maxSize: 250000,
        stepSize: 10
    });

    function coordChange(e: React.ChangeEvent<HTMLInputElement>){
        let value  = e.currentTarget.value.trim() === "" ? null : +e.currentTarget.value;
        const name = e.target.name as "x" | "y" | "z";

        setSeed((prev: { x: number | null; y: number | null; z: number | null }) => ({ ...prev, [name]: value }));
    }

    function paramsChange(e: React.ChangeEvent<HTMLInputElement>){
        const { name, value } = e.currentTarget;

        if (value == "")
            setParams(prev => ({
                ...prev,
                [name] : null
            }))

        else
            setParams(prev => ({
                ...prev,
                [name] : +value
            }))
    }

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>){
        e.preventDefault();
        const seed: number[] = (x !== null && y !== null && z !== null) ? [x, y, z]: [];
        const finalParams: Partial<ParamsProp> = {};
        (Object.keys(params) as (keyof ParamsProp)[]).forEach(key =>{
            if (params[key] !== null){
                finalParams[key] = params[key];
            }
        })

        const completeJson = {
            volume: volume,
            seed: seed,
            params: finalParams
        }

        try {
            const response = await fetch(`${API_BASE}/generate_segment`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(completeJson)
            });

            //console.log(JSON.stringify(completeJson))
            if (!response.ok)
                throw new Error(`Request failed with status ${response.status}`);

            const data = await response.json();
            console.log("Segmentation task started:", data);
            
            setSegments((prev: [string, number][]) => [...prev, [data.uuid, 0]]);
            pingJobStatus(data.uuid);
        }
        catch(error) {
            console.error("Failed to start segmentation:", error);
        }

        return;
    }

    function pingJobStatus(uuid: string) {
        const interval = setInterval(async () => {
            try {
            const res = await fetch(`${API_BASE}/jobs/${uuid}`);
            const status = await res.json();

            console.log("Job status:", status);

            if (status.state === "SUCCESS") {
                setSegments((prev: [string, number][]) => prev.map(([key, value]) => uuid === key ? [key, 1] : [key, value]));
                clearInterval(interval);
                console.log("Job finished:", status.result);
            }

            if (status.state === "FAILURE") {
                setSegments((prev: [string, number][]) => prev.filter(([key]) => key !== uuid));
                clearInterval(interval);
                console.error("Job failed:", status.error);
            }

            } catch (err) {
                setSegments((prev: [string, number][]) => prev.filter(([key]) => key !== uuid));
                clearInterval(interval);
                console.error("Polling failed:", err);
            }
        }, 2000);
    }

    return (
        <form onSubmit={handleSubmit}>
            <label>
                Seed
                <div className="seedRow">
                    <label>
                        x: 
                        <input name="x" type="number" onChange={coordChange} value={x !== null ? x : ""}/>
                    </label>
                    <label>
                        y: 
                        <input name="y" type="number" onChange={coordChange} value={y !== null ? y : ""}/>
                    </label>
                    <label>
                        z: 
                        <input name="z" type="number" onChange={coordChange} value={z !== null ? z : ""}/>
                    </label>
                    <button type="button" onClick={() => {
                        setIsFocusPoint(false);
                        setSeed({x: null, y: null, z: null});
                        }}>Reset focus point</button>
                </div>
            </label>
            <label>
                Additional Parameters
                <div>
                    <label>
                        Min Size: 
                        <input name="minSize" type="number" onChange={paramsChange} value={params.minSize !== null ? params.minSize : ""}/>
                    </label>
                    <label>
                        Max Size: 
                        <input name="maxSize" type="number" onChange={paramsChange} value={params.maxSize !== null ? params.maxSize : ""}/>
                    </label>
                    <label>
                        Step Size: 
                        <input name="stepSize" type="number" onChange={paramsChange} value={params.stepSize !== null ? params.stepSize : ""}/>
                    </label>
                </div>
                <div>
                    <label>
                        Global Threshold:
                        <input name="globalThreshold" type="number" onChange={paramsChange} value={params.globalThreshold !== null ? params.globalThreshold : ""} />
                    </label>
                    <label>
                        Allowed Difference:
                        <input name="allowedDifference" type="number" onChange={paramsChange} value={params.allowedDifference !== null ? params.allowedDifference : ""} />
                    </label>
                    <label>
                        Patience Max:
                        <input name="patienceMax" type="number" onChange={paramsChange} value={params.patienceMax !== null ? params.patienceMax : ""} />
                    </label>
                </div>
            </label>
            <button type="submit">Generate Segment</button>
        </form>
    );
}