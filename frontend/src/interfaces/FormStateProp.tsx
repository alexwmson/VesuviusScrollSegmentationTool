export interface ParamsProp{
    globalThreshold: number | null;
    allowedDifference: number | null;
    patienceMax: number | null;
    minSize: number | null;
    maxSize: number | null;
    stepSize: number | null;
}

export interface FormStateProp{
    volume: string;
    seed: number[];
    params: ParamsProp;
}
