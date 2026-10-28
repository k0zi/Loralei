# Loralei.DatasetMaker

A loralei GUI **Caption** oldalának motorja: egy könyvtár (rekurzívan) összes képéhez `<képneve>.txt`
leírást generál LoRA tanításhoz, egy OpenAI-kompatibilis vision végpont segítségével (OpenAI, LM Studio,
Ollama, vLLM, llama.cpp, OpenRouter...). Az alapértelmezett prompt természetes nyelvű, részletes
leírásokat kér (Z-Image, Qwen-Image, LTX).

Parancssori eszközként indult; ide könyvtárként került be. Nincs saját `Main`-je és nem ír a konzolra:
a felülete a `Loralei.Gui/ViewModels/CaptionViewModel.cs` + `Views/CaptionView.axaml`, a korábbi
konzolsorok pedig `ProcessingEvent`-ként érkeznek. A futás a GUI folyamatán belül megy, és az oldalon
látszik: haladásjelző, számlálók, eltelt/hátralévő idő, az éppen feldolgozott képek és a legutóbbi
leírás a kép bélyegképével. A közös kimeneti panelre csak az indítás, a hibák és az eredmény kerül.

## Használat

```bash
dotnet run --project Loralei.Gui -- --page Caption    # a gui/ mappából
dotnet test Loralei.DatasetMaker.Tests
```

Ami régen kapcsoló volt, az most mező az oldalon:

| Régi kapcsoló | Az oldalon |
|---|---|
| `<könyvtár>` | Dataset folder |
| `--trigger <szó>` | Trigger word – a leírás elejére kerül: `ohwx, ...` |
| `--context <szöveg>`, `--context-file <path>` | Extra context – a system prompt végére kerül |
| `--overwrite` | Overwrite existing captions (alapból kihagyja, aminek már van `.txt`-je; nem jegyzi meg) |
| `--parallel <n>` | Parallel requests |
| `--model <név>` | Model |
| `--dry-run` | Preview gomb – csak kilistázza, mit dolgozna fel, semmit nem küld el |
| `--config <path>` | nincs: a mezők maguk a config, lásd lent |

Támogatott formátumok: jpg, jpeg, png, webp, bmp, gif. Ha ugyanabban a mappában `foo.jpg` és `foo.png`
is van, a második kimarad (ugyanaz lenne a `.txt` neve).

A futás a GUI parancszárát használja (egyszerre egy dolog fut, mert a végpont jellemzően ugyanazon a
GPU-n lévő helyi modell), és a címsor Interrupt gombjával megszakítható; a már megírt leírások megmaradnak.

## Config

A mezők a `~/.config/loralei-gui/dataset-maker.json` fájlba mentődnek (a GUI saját `settings.json`-ja
mellé; `LORALEI_GUI_SETTINGS` ezt is áthelyezi), amikor elhagyod az oldalt, bezárod az ablakot vagy
futtatsz. A fájl alakja a régi `appsettings.json`-é, és csak a tulajdonos olvashatja (0600), mert API
kulcs lehet benne. Amíg nincs ilyen fájl, a kiindulás az exe mellé másolt `dataset-maker.defaults.json`.

- `OpenAI.BaseUrl` – pl. `https://api.openai.com/v1`, `http://localhost:1234/v1` (LM Studio), `http://localhost:11434/v1` (Ollama)
- `OpenAI.ApiKey` – ha üres, a `DATASETMAKER_API_KEY` vagy `OPENAI_API_KEY` env var-t használja (ezt futáskor
  olvassa ki, a fájlba nem írja be); ha az sincs, auth nélkül hív
- `OpenAI.Model`, `MaxTokens`, `Temperature`, `TimeoutSeconds`, `ImageDetail` (`low`/`high`/`auto`)
- `Caption.SystemPrompt` – a leírás stílusát meghatározó prompt (üresen a beépített alapértelmezés)
- `Caption.ExtraSystemPrompt`, `Caption.TriggerWord` – az Extra context / Trigger word mező
- `Caption.MaxParallel`, `Caption.MaxRetries` – párhuzamosság, újrapróbálkozások (429/5xx/timeout esetén exponenciális backoff)

## Felépítés

```
Loralei.DatasetMaker/
  Config/      AppConfig (a config alakja), ConfigStore (betöltés, mentés, API kulcs feloldása)
  Captioning/  OpenAiCaptionClient (/chat/completions), PromptBuilder (prompt és utófeldolgozás)
  Processing/  ImageScanner, DatasetProcessor (kihagyás, párhuzamosság, újrapróbálkozás, atomi írás)
Loralei.DatasetMaker.Tests/
```

A tesztek egy loopback porton futó ál-végponton (`TestSupport/StubEndpoint.cs`) hajtják át a valódi
HTTP klienst, így a tényleges kérést ellenőrzik, és semmi nem hagyja el a gépet.
