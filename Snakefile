rule Numerical_Varying_M:
    input:
        'src/scripts/numerical_M.py'
    output:
        'src/data/numerical_M.npz'
    conda:
        'environment.yml'
    script:
        "src/scripts/numerical_M.py"

rule Numerical_Varying_N:
    input:
        'src/scripts/numerical_N.py'
    output:
        'src/data/numerical_N.npz'
    conda:
        'environment.yml'
    script:
        "src/scripts/numerical_N.py"

rule Analytical_Varying_N:
    input:
        'src/scripts/analytical_N.py'
    output:
        'src/data/analytical_N.npz'
    conda:
        'environment.yml'
    script:
        "src/scripts/analytical_N.py"