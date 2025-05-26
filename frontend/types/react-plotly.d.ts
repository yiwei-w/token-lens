declare module 'react-plotly.js' {
  import * as React from 'react';
  
  interface PlotParams {
    data?: any[];
    layout?: any;
    frames?: any[];
    config?: any;
    style?: React.CSSProperties;
    useResizeHandler?: boolean;
    revision?: number;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    onPurge?: (figure: any, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
    divId?: string;
    className?: string;
    debug?: boolean;
    [key: string]: any;
  }

  class Plot extends React.Component<PlotParams> {}
  
  export default Plot;
} 