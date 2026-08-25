import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const sourcePptx = "C:/Users/ranic/OneDrive - George Mason University - O365 Production/04_2026 Spring/BINF751 - Biochem Modeling/Final Project/presntation/BINF751_finalproject_RVA_v8.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(sourcePptx));
const image = presentation.resolve("im/0vud8v2d");
console.log(Object.getOwnPropertyNames(Object.getPrototypeOf(image)).sort());
