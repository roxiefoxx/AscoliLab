import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePptx = "C:/Users/ranic/OneDrive - George Mason University - O365 Production/04_2026 Spring/BINF751 - Biochem Modeling/Final Project/presntation/BINF751_finalproject_RVA_v8.pptx";
const outputDir = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/pptx_build/source_render";
await fs.mkdir(outputDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePptx));
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(`${outputDir}/${stem}.png`, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(`${outputDir}/${stem}.layout.json`, await layout.text());
}
const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${outputDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));
console.log(`exported ${presentation.slides.items.length} slides`);
