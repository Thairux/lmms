/*
 * AIStudioView.h - UI for the AIStudio tool plugin
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

#ifndef LMMS_GUI_AI_STUDIO_VIEW_H
#define LMMS_GUI_AI_STUDIO_VIEW_H

#include <QProcess>

#include "ToolPluginView.h"

class QComboBox;
class QLineEdit;
class QTextEdit;
class QPushButton;
class QLabel;

namespace lmms
{
class AIStudio;
}

namespace lmms::gui
{

class AIStudioView : public ToolPluginView
{
	Q_OBJECT

public:
	explicit AIStudioView(AIStudio* plugin);
	~AIStudioView() override = default;

private slots:
	void generate();
	void requestFinished(int exitCode);
	void requestErrored(QProcess::ProcessError);

private:
	QString resolveApiKey() const;
	QString selectedProviderName() const;
	QByteArray buildRequestBody(const QString& prompt) const;
	QString parseResponse(const QByteArray& responseBody) const;
	void setBusy(bool busy);

	AIStudio* m_plugin = nullptr;
	QComboBox* m_provider = nullptr;
	QLineEdit* m_apiKey = nullptr;
	QTextEdit* m_prompt = nullptr;
	QTextEdit* m_response = nullptr;
	QPushButton* m_generateButton = nullptr;
	QLabel* m_status = nullptr;
	QProcess* m_process = nullptr;
};

} // namespace lmms::gui

#endif // LMMS_GUI_AI_STUDIO_VIEW_H
