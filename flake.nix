{
	description = "A flake for building wodan-rs";

	inputs = {
		#nixpkgs.url = "nixpkgs/nixpkgs-unstable";
		nixpkgs.url = "nixpkgs";
		llama.url = "github:ggerganov/llama.cpp/b5ffb2849d23afe73647f68eec7b68187af09be6";
	};

	outputs = { self, nixpkgs, llama }:
	let
		pkgs = import nixpkgs {
			system = "x86_64-linux";
			config = {
				allowUnfree = true;
				cudaSupport = true;
				permittedInsecurePackages = [
					"openssl-1.1.1u"
				];
			};
		};
		macpkgs = import nixpkgs {
			system = "x86_64-darwin";
			config = {allowUnfree = true;};
		};

		py-pkgs = pp: with pp; [
			pip
		];
		my-py = pkgs.python3Full.withPackages py-pkgs;
		mac-py = macpkgs.python3Full.withPackages py-pkgs;
		mac_fks = macpkgs.darwin.apple_sdk.frameworks;

		cuda-fhs = pkgs.buildFHSUserEnv {
			name = "cuda-env";
			targetPkgs = pkgs: with pkgs; [ 
				git
				gnupg
				autoconf
				curl
				procps
				gnumake
				gcc
				utillinux
				m4
				gperf
				unzip
				libGLU
				libGL
				xorg.libXi 
				xorg.libXmu 
				xorg.libXext 
				xorg.libX11
				xorg.libXv 
				xorg.libXrandr
				zlib
				ncurses5
				stdenv.cc
				binutils
				pkg-config
				libconfig
				cmake
				my-py
				cudaPackages.cudatoolkit
				linuxPackages.nvidia_x11
				# Libfacedetection
				glib
				opencv
				#libtorch-bin
				wgpu-utils
				vulkan-tools
				vulkan-loader
				vulkan-headers
				cudaPackages.cudnn
				vulkan-validation-layers
			];
			multiPkgs = pkgs: with pkgs; [ zlib ];
			runScript = "bash";
			profile = ''
				export PIP_PREFIX="$(pwd)/_build/pip_packages"
				export PYTHONPATH="$PIP_PREFIX/${pkgs.python3Full.sitePackages}:$PYTHONPATH"
				export DEBUGPYPATH="${my-py}/bin/python3"
				export VIRTUAL_ENV="$(pwd)/.venv"
				export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
				export PATH=$CUDA_PATH:$PATH
				unset SOURCE_DATE_EPOCH
			'';
		};
		llama-pkg = llama.packages.x86_64-linux.default.overrideAttrs (finalAttrs: previousAttrs: {
			nativeBuildInputs = previousAttrs.nativeBuildInputs ++ [
				pkgs.openblas
				pkgs.cudaPackages.cudatoolkit
			];
			cmakeFlags = previousAttrs.cmakeFlags ++ [
				"-DLLAMA_CUBLAS=ON"
			];
		});
	in {

		devShell.x86_64-linux = pkgs.mkShell {
			nativeBuildInputs = with pkgs; [ 
				cuda-fhs
				my-py
				cudaPackages.cudatoolkit
				cudaPackages.cudnn
				linuxPackages.nvidia_x11
				openblas
				llama-pkg
				#pylyzer
				ruff
				ruff-lsp
			];
			LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH";
			CUDA_PATH = "${pkgs.cudatoolkit}";
			EXTRA_CCFLAGS = "-I/usr/include";
			EXTRA_LDFLAGS = "-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
			#shellHook = ''exec cuda-env'';
		};
	};
}
