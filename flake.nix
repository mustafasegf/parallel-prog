{
  description = "MPI devshell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in with pkgs; {
        devShells.default = mkShell {
          nativeBuildInputs = [ cmake gnumake pkg-config ];
          buildInputs = [ openmpi cudaPackages.cudatoolkit ];

          packages = with pkgs; [ gcc12 ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.zluda}/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.rocmPackages.clr}/lib/:$LD_LIBRARY_PATH
            # export ROCM_VISIBLE_DEVICES=GPU-a8ec4909d84b514e
            # export HIP_VISIBLE_DEVICES=0
          '';
        };
      });
}
