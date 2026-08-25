import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const FINAL_PPTX = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/BINF751_finalproject_RVA_results_v9.pptx";
const OUT_DIR = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/pptx_build/final_render_all";
await fs.mkdir(OUT_DIR, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(FINAL_PPTX));
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${OUT_DIR}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${OUT_DIR}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));
console.log(`rendered ${presentation.slides.items.length} slides`);
