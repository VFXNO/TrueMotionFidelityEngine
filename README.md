# True Motion Fidelity Engine (WGC + Shader Interpolation)

Windows 11 frame generation prototype using Windows Graphics Capture, Direct3D 11 compute shaders, and ImGui.

## Build

- Requires Visual Studio 2022 with Desktop C++ workload
- Windows 11 SDK (10.0.26100 or newer recommended)

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Run the generated `True Motion Fidelity Engine.exe` from `build/bin/Release`.


## License & Branding

Distributed under the Apache License 2.0. See [`LICENSE`](LICENSE) for more information.

**Important Note on Branding:**
The name "True Motion Fidelity Engine" is a trademark of VFXNO.
- You **CAN** fork this repository and modify the code.
- You **CAN** use this code commercially.
- You **CANNOT** use the name "True Motion Fidelity Engine" or the project's logos in your derivative works without permission.
- If you distribute a modified version, you must rename it.
- You must include the `NOTICE` file in your distribution, keeping the attribution "Powered by VFXNO".
