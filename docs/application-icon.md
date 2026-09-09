# Application icon

`logo/astra_icon.svg` is the source artwork. `build/icon.rs` renders it during
the Cargo build, preserving its proportions on a transparent square. The SVG
renderer is a build dependency only; it is not linked into the application.
Changing the SVG automatically regenerates the icons on the next build.

Windows builds embed a multi-resolution ICO resource in the executable (16–256
pixels). The window also uses an embedded 64px RGBA icon on platforms supported
by winit. No external logo files are needed at runtime. Windows resource
compilation requires the Windows SDK resource compiler on MSVC, or windres for
GNU targets.

On macOS, Finder and Dock use the application bundle's ICNS resource. To build
and package the release application, run:

```sh
python3 scripts/package_macos.py
```

The script builds the binary, copies the generated ICNS into
`target/release/Astra.app`, writes its Info.plist and applies an ad-hoc signature.
Open or distribute `Astra.app` to get the application icon and launch without a
terminal. A standalone Unix executable is not an application bundle.

The packaging script does not provide a Developer ID signature or notarization.
