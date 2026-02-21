/*
 * AIStudio.cpp - Tool plugin for AI-assisted composition and mix ideation
 *
 * Copyright (c) 2026
 *
 * This file is part of LMMS - https://lmms.io
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program (see COPYING); if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301 USA.
 *
 */

#include "AIStudio.h"

#include <QObject>

#include "AIStudioView.h"
#include "plugin_export.h"

namespace lmms
{

extern "C"
{

Plugin::Descriptor PLUGIN_EXPORT aistudio_plugin_descriptor =
{
	LMMS_STRINGIFY(PLUGIN_NAME),
	"AI Studio",
	QT_TRANSLATE_NOOP("PluginBrowser",
		"Generate arrangement and mix ideas using free-tier AI providers"),
	"LMMS community",
	0x0100,
	Plugin::Type::Tool,
	nullptr,
	nullptr,
	nullptr,
} ;

PLUGIN_EXPORT Plugin* lmms_plugin_main(Model*, void*)
{
	return new AIStudio;
}

}


AIStudio::AIStudio() :
	ToolPlugin(&aistudio_plugin_descriptor, nullptr)
{
}


gui::PluginView* AIStudio::instantiateView(QWidget*)
{
	return new gui::AIStudioView(this);
}


QString AIStudio::nodeName() const
{
	return aistudio_plugin_descriptor.name;
}


QString AIStudio::localFallbackResponse(const QString& prompt)
{
	const QString base = prompt.trimmed().isEmpty() ? QStringLiteral("untitled idea") : prompt.trimmed();
	return QObject::tr(
		"Local fallback mode (free, no API key).\n\n"
		"Prompt: %1\n\n"
		"Arrangement blueprint:\n"
		"- Intro: 8 bars, low-pass filtered main chord stack\n"
		"- Drop A: 16 bars, sidechained bass + layered transient click\n"
		"- Break: 8 bars, remove kick, automate stereo width on pads\n"
		"- Drop B: 16 bars, octave bass reinforcement and extra percussion\n\n"
		"Mix chain suggestion:\n"
		"- Kick: high-pass at 25Hz, narrow boost near 60Hz, 2-3 dB clipper ceiling\n"
		"- Bass bus: dynamic EQ around 180-300Hz keyed from kick\n"
		"- Lead bus: de-esser around 6-8kHz, stereo widener after EQ\n"
		"- Master: soft clipper before limiter, target integrated loudness -9 to -7 LUFS\n\n"
		"Realtime AI hard-path next step:\n"
		"Run an external local audio model service and bridge it with block audio I/O.")
		.arg(base);
}

} // namespace lmms
