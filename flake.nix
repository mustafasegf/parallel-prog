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
            export LD_LIBRARY_PATH=/nix/store/nlw286w6v81r3imr61xkyq52wv0l20d7-zluda-3/lib:$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=/nix/store/shr03swgapa05gsh7y0zfch2vs3xb7s7-clr-6.0.2/lib/:$LD_LIBRARY_PATH
            # export ROCM_VISIBLE_DEVICES=GPU-a8ec4909d84b514e
            # export HIP_VISIBLE_DEVICES=0
          '';
        };
      });
}
