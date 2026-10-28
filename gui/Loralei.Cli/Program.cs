using System.Runtime.InteropServices;
using Loralei.Cli;
using Loralei.Core.Config;
using Loralei.Core.Dto;
using Loralei.Core.Models;

// Ctrl-C (and the GUI's Interrupt, which sends the same signal) asks the running job to
// stop cleanly instead of killing this process: the worker needs the chance to stamp its
// manifest and release the block swapper, and this process has to stay to see it through.
using var cancellation = new CancellationTokenSource();
using var sigint = PosixSignalRegistration.Create(PosixSignal.SIGINT, context =>
{
    context.Cancel = true;
    cancellation.Cancel();
});
using var sigterm = PosixSignalRegistration.Create(PosixSignal.SIGTERM, context =>
{
    context.Cancel = true;
    cancellation.Cancel();
});

string? usage = null;
try
{
    if (args.Length == 0 || args[0] is "--help" or "-h" or "help")
    {
        Terminal.Line(Help.Root);
        return 0;
    }
    if (args[0] == "--version")
    {
        Terminal.Line($"loralei {LoraleiInfo.Version}");
        return 0;
    }

    var rest = args[1..];
    switch (args[0])
    {
        case "doctor": usage = Help.Doctor; return await WorkerCommands.Doctor(rest, cancellation.Token);
        case "cache": usage = Help.Cache; return await WorkerCommands.Cache(rest, cancellation.Token);
        case "train": usage = Help.Train; return await WorkerCommands.Train(rest, cancellation.Token);
        case "sample": usage = Help.Sample; return await WorkerCommands.Sample(rest, cancellation.Token);
        case "export": usage = Help.Export; return await WorkerCommands.Export(rest, cancellation.Token);
        case "models": usage = Help.Models; return await LocalCommands.Models(rest, cancellation.Token);
        case "data": usage = Help.Data; return LocalCommands.Data(rest);
        case "runs": usage = Help.Runs; return LocalCommands.Runs(rest);
        default:
            usage = Help.Root;
            throw new UsageException($"no such command '{args[0]}'");
    }
}
catch (HelpRequested help)
{
    Terminal.Line(help.Text);
    return 0;
}
catch (UsageException e)
{
    Terminal.Line(Terminal.Red($"error: {e.Message}"));
    if (usage is not null) Terminal.Line(usage);
    return 2;
}
catch (Exception e) when (e is ConfigException or NotEnoughSpaceException or HttpRequestException
                              or IOException or UnauthorizedAccessException or InvalidDataException)
{
    Terminal.Line(Terminal.Red(e.Message));
    return 1;
}
catch (OperationCanceledException)
{
    Terminal.Line(Terminal.Yellow("interrupted"));
    return 130;
}

namespace Loralei.Cli
{
    /// <summary>Thrown by a command that saw <c>--help</c>, carrying its usage text.</summary>
    public sealed class HelpRequested(string text) : Exception
    {
        public string Text { get; } = text;
    }

    public static class Help
    {
        public const string Root = """
            loralei - ROCm-only LoRA trainer for Qwen-Image, Krea 2 and LTX 2.3.

            usage: loralei <command> [options]

              doctor              validate the ROCm stack; exits non-zero if anything is broken
              models ls|pull|gc   inspect, fetch and reclaim base model weights
              data check <dir>    pair images with captions and preview the buckets
              cache <dir>         pre-compute VAE latents and text embeddings
              train <dir>         train a LoRA on a directory of images and captions
              sample              render images from a base model, optionally with a LoRA
              export <run>        write the LoRA in diffusers and ComfyUI formats
              runs ls|show|rm     manage training runs

            `loralei <command> --help` lists a command's options. --version prints the version.
            """;

        public const string Doctor = """
            usage: loralei doctor [--json] [--fix] [--thorough]
                   loralei doctor --install     set up the Python worker: .venv, ROCm torch, loralei
            """;

        public const string Models = """
            usage: loralei models ls   [--models-dir PATH] [--model-path PATH] [--json]
                   loralei models pull <model>... [--models-dir PATH] [--model-path PATH] [--dry-run]
                   loralei models gc   [--dry-run | --no-dry-run] [--min-gib N] [--yes] [--json]
            """;

        public const string Data =
            "usage: loralei data check <dir> [-m MODEL] [--resolution N] [--caption-extension .txt] [--json]";

        public const string Cache = """
            usage: loralei cache <dir> [-m MODEL] [--resolution N] [--gpu N] [--vae-device cpu|cuda]
                                 [--models-dir PATH] [--model-path PATH] [--force]
            """;

        public const string Train = """
            usage: loralei train <dir> --name NAME [options]

              -m, --model MODEL        qwen-image (default), krea-2-raw, ltx-2.3
              --profile NAME           balanced (default), quality, speed, lowmem
              --rank N  --lr X  --steps N  --batch-size N  --seed N
              --resolution N           a multiple of 16, or of 32 for ltx-2.3
              --blocks-to-swap N       frozen blocks parked in host RAM
              --fp8-base | --no-fp8-base
              --optimizer NAME         adamw8bit (default), adamw, adafactor
              --trigger WORD           prepend a trigger word to captions
              --vae-device cpu|cuda    cpu is the safe default
              --gpu N                  which card to train on
              --save-every N           checkpoint interval (0 = only at the end)
              --models-dir PATH  --model-path PATH
              --resume                 continue from the newest checkpoint
              --dry-run                resolve, estimate, stop

            Settings without a flag are read from ./loralei.yaml.
            """;

        public const string Sample = """
            usage: loralei sample -p PROMPT [-p PROMPT ...] [--lora FILE] [-m MODEL] [-o DIR]
                                  [--steps N] [--cfg X] [--width N] [--height N] [--seed N] [--gpu N]
                                  [--models-dir PATH] [--model-path PATH]
            """;

        public const string Export = """
            usage: loralei export <run | file.safetensors> [--format comfyui|diffusers|both]
                                  [--output-root DIR] [--verify | --no-verify]
            """;

        public const string Runs = """
            usage: loralei runs ls   [--output-root DIR]
                   loralei runs show <name> [--output-root DIR]
                   loralei runs rm   <name> [--output-root DIR] [--yes]
            """;
    }
}
