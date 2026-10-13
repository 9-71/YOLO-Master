import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { JobProvider } from "./state/JobContext";

createRoot(document.getElementById("root")!).render(<StrictMode><JobProvider><App /></JobProvider></StrictMode>);
