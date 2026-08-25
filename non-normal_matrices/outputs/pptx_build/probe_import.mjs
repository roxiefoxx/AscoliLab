import fs from "node:fs/promises";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePptx = "C:/Users/ranic/OneDrive - George Mason University - O365 Production/04_2026 Spring/BINF751 - Biochem Modeling/Final Project/presntation/BINF751_finalproject_RVA_v8.pptx";
const outputDir = "C:/Users/ranic/Documents/GitHub/non-normal_matrices/outputs/pptx_build/probe";
await fs.mkdir(outputDir, { recursive: true });

const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePptx));
const inspect = await presentation.inspect({
  kind: "slide,textbox,shape,image,table,chart,notes,layout",
  include: "id,slide,name,title,textPreview,bbox,isPlaceholder",
  maxChars: 20000,
});
await fs.writeFile(`${outputDir}/inspect.ndjson`, inspect.ndjson);

const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(`${outputDir}/montage.webp`, new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(`${outputDir}/roundtrip.pptx`);

console.log(`slides=${presentation.slides.items.length}`);
