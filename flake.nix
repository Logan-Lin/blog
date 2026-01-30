{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
  in {
    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          zola
          imagemagick
          (writeShellScriptBin "serve" ''
            zola serve --open
          '')
          (writeShellScriptBin "build" ''
            zola build
          '')
          (writeShellScriptBin "compress-images" ''
            if [ -z "$1" ]; then
              echo "Usage: compress-images <folder>"
              exit 1
            fi
            find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | while read -r img; do
              outfile="''${img%.*}.webp"
              ${imagemagick}/bin/magick "$img" -resize '1800>' -quality 82 "$outfile"
              echo "Converted: $img -> $outfile"
            done
          '')
        ];
        shellHook = ''
          echo "Zola blog development environment"
          echo "Commands: serve, build, zola"
        '';
      };
    });
  };
}
