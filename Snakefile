configfile: "workflow/config/params.yaml"
include: "workflow/rules/gemm.smk"

defaults = config["gemm"]["defaults"]
shapes = config["gemm"]["shapes"]
stream_hw_id = defaults["stream_hw_identifier"]
trace_size = defaults["trace_size"]

rule all:
    input:
        expand(
            "outputs/{stream_hw_id}-gemm_{M}_{K}_{N}-fused-constraint-optimization/status.ok",
            stream_hw_id=stream_hw_id,
            trace_size=trace_size,
            M=[shape["M"] for shape in shapes],
            K=[shape["K"] for shape in shapes],
            N=[shape["N"] for shape in shapes],
        )